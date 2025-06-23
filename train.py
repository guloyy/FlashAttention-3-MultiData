import os
import time
import copy
import gc
import csv
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs, Attention
from utils import (
    build_lr_scheduler,
    clip_grad_norm_,
    get_args,
    get_num_params,
    get_num_flop_per_token,
    init_logger,
    logger,
    PRECISION_STR_TO_DTYPE,
    set_default_dtype,
    sdp_kernel
)


def save_speed_result(model_type: str, avg_time_ms: float, path="speed_results.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    results[model_type] = avg_time_ms

    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_speed_result(path="speed_results.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def measure_speedup(model, dataloader, device, steps=50):
    model.eval()
    input_ids, _ = next(iter(dataloader))
    input_ids = input_ids.to(device)

    # Warmup
    for _ in range(5):
        model(input_ids)

    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(steps):
            model(input_ids)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    avg_time_ms = elapsed_ms / steps

    mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    logger.info(f"[Speed] Avg inference time: {avg_time_ms:.2f} ms | Max GPU memory: {mem_used:.2f} GB")
    return avg_time_ms


def train(args):
    logger.info(f"Experiment args: {args}")
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

    logger.info("Setting up DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_ds = ParquetDataset(
        args.dataset,
        tokenizer,
        args.sequence_length,
        args.batch_size * args.training_steps
    )
    train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=train_collator)
    train_dl_iterator = iter(train_dl)

    logger.info("Setting up Model...")
    model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )

    with set_default_dtype(model_dtype):
        model = Transformer(model_config, use_flash_attn=args.use_flash_attn).to(device)

    if args.compile and not any(isinstance(m, Attention) for m in model.modules()):
        logger.info("Using torch.compile")
        model = torch.compile(model, fullgraph=True)
    else:
        logger.warning("⚠️ Skipping torch.compile because FlashAttention3 is incompatible with TorchDynamo.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        model_config,
    )

    ntokens_since_last_log = 0
    ntraining_tokens_since_last_log = 0
    time_last_log = time.perf_counter()

    metrics_file = open("training_metrics.csv", "w", newline="")
    csv_writer = csv.writer(metrics_file)
    csv_writer.writerow(["step", "tokens_per_sec", "train_token_pct", "mfu_pct", "tflops"])

    tps_history = []
    train_pct_history = []
    mfu_history = []
    tflops_history = []

    logger.info(f"Running with {'FlashAttention' if args.use_flash_attn else 'standard attention'}")
    logger.info("Starting training!")

    total_start_time = time.perf_counter()
    model.train()

    for train_step in range(1, args.training_steps + 1):
        input_ids, labels = next(train_dl_iterator)
        ntokens_since_last_log += args.batch_size * args.sequence_length
        num_items_in_batch = labels.ne(-100).sum()
        ntraining_tokens_since_last_log += num_items_in_batch
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum"
        )
        loss /= num_items_in_batch
        loss.backward()

        clip_grad_norm_(model.parameters(), args.grad_max_norm)
        optimizer.step()
        lr_scheduler.step()

        if train_step % args.logging_frequency == 0:
            time_delta = time.perf_counter() - time_last_log
            tps = ntokens_since_last_log / time_delta
            training_tps = ntraining_tokens_since_last_log / time_delta
            train_pct = 100 * training_tps / tps
            mfu = 100 * num_flop_per_token * tps / 989e12
            tflops = num_flop_per_token * tps / 1e12

            csv_writer.writerow([train_step, tps, train_pct, mfu, tflops])

            tps_history.append(tps)
            train_pct_history.append(train_pct)
            mfu_history.append(mfu)
            tflops_history.append(tflops)

            logger.info(
                f"Step: {train_step} | Loss: {loss:.2f} | "
                f"Tokens/sec: {tps:.2f} | Training tokens/sec (%): {train_pct:.2f} | "
                f"MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}"
            )
            ntokens_since_last_log = 0
            ntraining_tokens_since_last_log = 0
            time_last_log = time.perf_counter()

    metrics_file.close()

    logger.info("Training completed")

    if tps_history:
        avg_tps = sum(tps_history) / len(tps_history)
        avg_pct = sum(train_pct_history) / len(train_pct_history)
        avg_mfu = sum(mfu_history) / len(mfu_history)
        avg_tflop = sum(tflops_history) / len(tflops_history)
        logger.info("=== Average performance over training ===")
        logger.info(f"  Avg Tokens/sec:      {avg_tps:.2f}")
        logger.info(f"  Avg Train Token %:   {avg_pct:.2f}")
        logger.info(f"  Avg MFU %:           {avg_mfu:.2f}")
        logger.info(f"  Avg TFLOPs:          {avg_tflop:.2f}")

    total_time = time.perf_counter() - total_start_time
    logger.info(f"Total training time: {total_time:.2f} seconds")

    return model, tokenizer, train_dl, device, model_config


if __name__ == "__main__":
    init_logger()
    args = get_args()
    model, tokenizer, train_dl, device, model_config = train(args)

    logger.info("Measuring speed of current model...")
    avg_time_ms = measure_speedup(model, train_dl, device)
    if args.save_speed_key:
        save_speed_result(args.save_speed_key, avg_time_ms, path=args.speed_results_path)

    all_results = load_speed_result(path=args.speed_results_path)
    if "flash" in all_results and "baseline" in all_results:
        baseline = all_results["baseline"]
        flash = all_results.get("flash", avg_time_ms)
        speedup = baseline / flash
        logger.info("\n[SPEEDUP COMPARISON]")
        logger.info(f"{'Model':<12} | {'Avg Time (ms)':>15}")
        logger.info(f"{'Baseline':<12} | {baseline:>15.2f}")
        logger.info(f"{'Flash':<12}   | {flash:>15.2f}")
        logger.info(f"Speedup: {speedup:.2f}x")






