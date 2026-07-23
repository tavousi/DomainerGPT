#!/usr/bin/env python3
"""
generate_domains.py — Generate names and safely check availability with Smart Delays.
"""

import os
import pickle
import socket
import argparse
import concurrent.futures
import time
import numpy as np

from data import load_dataset
from backends.numpy_backend import MicroGPTNumpy

MODEL_CACHE_FILE = "microgpt_model_cache.pkl"

def parse_args():
    p = argparse.ArgumentParser(description="Generate names and check domain availability safely")
    p.add_argument("--count",       type=int,   default=1000, help="تعداد دامنه‌های یکتایی که باید تولید شود")
    p.add_argument("--tld",         type=str,   default="com", help="پسوند دامنه (مثل com, ir, net)")
    p.add_argument("--threads",     type=int,   default=15,   help="تعداد نخ‌های همزمان (پیشنهاد برای مودم خانگی: ۱۰ تا ۲۰)")
    p.add_argument("--temperature", type=float, default=0.7,  help="میزان خلاقیت مدل")
    p.add_argument("--chunk_size",  type=int,   default=300,  help="تعداد دامنه‌ها در هر دسته قبل از استراحت")
    p.add_argument("--sleep_time",  type=float, default=5.0,  help="زمان استراحت بین هر دسته به ثانیه")
    return p.parse_args()

def check_domain_availability(domain):
    """
    بررسی سریع آزاد بودن دامنه با استفاده از درخواست DNS.
    """
    try:
        socket.gethostbyname(domain)
        return domain, False # دامنه IP دارد (ثبت شده است)
    except socket.gaierror:
        return domain, True  # دامنه IP ندارد (احتمالاً آزاد است)
    except Exception:
        return domain, False

def main():
    args = parse_args()

    if not os.path.exists(MODEL_CACHE_FILE):
        print(f"خطا: فایل مدل '{MODEL_CACHE_FILE}' پیدا نشد. لطفاً ابتدا آموزش را اجرا کنید.")
        return

    print("در حال بارگذاری معماری و دایره لغات...")
    _, _, decode, BOS, vocab_size = load_dataset(seed=42)

    print("در حال بارگذاری وزن‌های شبکه عصبی آموزش‌دیده...")
    with open(MODEL_CACHE_FILE, "rb") as f:
        cache_data = pickle.load(f)

    cfg = cache_data["cfg"]
    model = MicroGPTNumpy(**cfg, seed=42)
    model.p = cache_data["weights"]
    print("مدل با موفقیت بارگذاری شد!\n")

    print(f"در حال تولید {args.count} نام یکتا...")
    generated_names = set()
    
    while len(generated_names) < args.count:
        ids = model.generate(BOS, vocab_size, temperature=args.temperature)
        name = decode(ids)
        if 4 <= len(name) <= 12 and name not in generated_names:
            generated_names.add(name)

    domains_to_check = [f"{name}.{args.tld.strip('.')}" for name in generated_names]
    available_domains = []

    print(f"\nشروع بررسی {len(domains_to_check)} دامنه.")
    print(f"تنظیمات شبکه: {args.threads} درخواست همزمان | استراحت {args.sleep_time} ثانیه‌ای پس از هر {args.chunk_size} درخواست.\n")

    t0 = time.time()
    
    # تقسیم لیست دامنه‌ها به دسته‌های کوچکتر (Chunks)
    chunks = [domains_to_check[i:i + args.chunk_size] for i in range(0, len(domains_to_check), args.chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        for index, chunk in enumerate(chunks):
            # بررسی دامنه‌های این دسته
            results = executor.map(check_domain_availability, chunk)
            
            for domain, is_available in results:
                if is_available:
                    available_domains.append(domain)
                    print(f"[+] آزاد: {domain}")
            
            # اعمال وقفه بین دسته‌ها (به جز دسته آخر)
            if index < len(chunks) - 1:
                print(f"\n[!] بررسی {args.chunk_size} دامنه تمام شد. {args.sleep_time} ثانیه استراحت برای جلوگیری از فشار به مودم...")
                time.sleep(args.sleep_time)

    t1 = time.time()
    
    output_file = f"available_domains_{args.tld}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for dom in available_domains:
            f.write(dom + "\n")

    print("\n" + "━"*50)
    print("گزارش نهایی:")
    print(f"تعداد کل بررسی شده: {args.count}")
    print(f"تعداد دامنه‌های آزاد یافت شده: {len(available_domains)}")
    print(f"زمان کل (شامل استراحت‌ها): {t1 - t0:.2f} ثانیه")
    print(f"لیست دامنه‌های آزاد در فایل '{output_file}' ذخیره شد.")
    print("━"*50)

if __name__ == "__main__":
    main()