#!/usr/bin/env python3
"""
benchmark.py — Optimized for Dataset #2, Numpy Only, with Batching and Caching.
"""

import argparse
import random
import time
import os
import pickle
import numpy as np

from data import load_dataset
from backends.numpy_backend import MicroGPTNumpy

# نام فایلی که مدل و پارامترها در آن ذخیره می‌شوند
MODEL_CACHE_FILE = "microgpt_model_cache.pkl"

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="microgpt benchmark (numpy only, with caching)")
    p.add_argument("--steps",      type=int,   default=30000, help="training steps")
    p.add_argument("--batch_size", type=int,   default=32,    help="batch size for training")
    p.add_argument("--n_embd",     type=int,   default=64)
    p.add_argument("--n_layer",    type=int,   default=2)
    p.add_argument("--n_head",     type=int,   default=4)
    p.add_argument("--block_size", type=int,   default=24,    help="max_length 20 + 2 BOS tokens")
    p.add_argument("--lr",         type=float, default=0.002)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--samples",    type=int,   default=10,    help="names to generate")
    p.add_argument("--log_every",  type=int,   default=1000,  help="print progress every N steps")
    return p.parse_args()


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, seed: int):
    return MicroGPTNumpy(**cfg, seed=seed)


# ── training loop ─────────────────────────────────────────────────────────────

def run_training(model, docs, encode, n_steps, lr, log_every, batch_size):
    losses     = []
    step_times = []

    for step in range(n_steps):
        lr_t   = lr * (1 - step / n_steps)
        
        # انتخاب تصادفی یک بچ از کلمات برای این مرحله
        batch_docs = random.sample(docs, batch_size)
        batch_tokens = [encode(doc) for doc in batch_docs]

        t0   = time.perf_counter()
        loss = model.train_step(batch_tokens, lr_t)
        t1   = time.perf_counter()

        losses.append(loss)
        step_times.append(t1 - t0)

        if (step + 1) % log_every == 0 or step == 0:
            ms = (t1 - t0) * 1000
            print(f"  step {step+1:5d}/{n_steps}  |  loss {loss:.4f}  |  {ms:8.2f} ms")

    return {
        "final_loss":  losses[-1],
        "avg_loss":    float(np.mean(losses[-100:])),
        "total_s":     sum(step_times),
        "median_ms":   float(np.median(step_times)) * 1000,
        "mean_ms":     float(np.mean(step_times))   * 1000,
    }


# ── results table ─────────────────────────────────────────────────────────────

def print_summary(result):
    print()
    print("=" * 64)
    print("  BENCHMARK SUMMARY (NumPy Only)")
    print("=" * 64)
    header = f"{'Backend':<14}  {'Final Loss':<11}  {'Avg(last100)':<13}  {'Total(s)':<10}  {'Median ms':<10}"
    print(header)
    print("-" * 64)
    print(
        f"  {'numpy':<12}  {result['final_loss']:<11.4f}  {result['avg_loss']:<13.4f}"
        f"  {result['total_s']:<10.2f}  {result['median_ms']:<10.2f}"
    )
    print("=" * 64)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset …")
    docs, encode, decode, BOS, vocab_size = load_dataset(seed=args.seed)
    print(f"  {len(docs)} docs  |  vocab size {vocab_size}")

    cfg = dict(
        vocab_size  = vocab_size,
        n_embd      = args.n_embd,
        n_layer     = args.n_layer,
        n_head      = args.n_head,
        block_size  = args.block_size,
    )
    
    train_cfg = dict(
        steps      = args.steps,
        batch_size = args.batch_size,
        lr         = args.lr,
        seed       = args.seed
    )

    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  Backend: numpy")
    print(f"{bar}")

    try:
        model = build_model(cfg, args.seed)
        
        trained = False
        # بررسی وجود فایل کش مدل
        if os.path.exists(MODEL_CACHE_FILE):
            try:
                with open(MODEL_CACHE_FILE, "rb") as f:
                    cache_data = pickle.load(f)
                # بررسی اینکه آیا معماری مدل و تنظیمات آموزش تغییر نکرده باشد
                if cache_data.get("cfg") == cfg and cache_data.get("train_cfg") == train_cfg:
                    print("\n[INFO] این مدل با همین پارامترها و داده‌ها از قبل آموزش دیده است.")
                    print("[INFO] در حال بارگذاری وزن‌های ذخیره شده به جای آموزش مجدد...\n")
                    model.p = cache_data["weights"]
                    trained = True
            except Exception as e:
                print(f"خطا در خواندن فایل ذخیره شده: {e}")
        
        if not trained:
            print("\n  در حال آموزش مدل جدید...")
            result = run_training(
                model, docs, encode,
                n_steps=args.steps,
                lr=args.lr,
                log_every=args.log_every,
                batch_size=args.batch_size
            )
            print_summary(result)
            
            # ذخیره مدل پس از پایان آموزش
            with open(MODEL_CACHE_FILE, "wb") as f:
                pickle.dump({
                    "cfg": cfg,
                    "train_cfg": train_cfg,
                    "weights": model.p
                }, f)
            print(f"[INFO] شبکه عصبی با موفقیت در فایل '{MODEL_CACHE_FILE}' ذخیره شد.")

        if args.samples > 0:
            print(f"\n  Generated names ({args.samples} samples, T=0.7):")
            for _ in range(args.samples):
                ids  = model.generate(BOS, vocab_size, temperature=0.7)
                name = decode(ids)
                print(f"    {name}")

    except RuntimeError as exc:
        print(f"  SKIPPED — {exc}")
        return

if __name__ == "__main__":
    main()
