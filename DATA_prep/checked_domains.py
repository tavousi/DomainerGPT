import asyncio
import aiodns
import pandas as pd
import os
import time
from pathlib import Path

# =========================
# تنظیمات
# =========================
INPUT_FILE = r"C:\Users\tavou\Downloads\Compressed\top-1m\domains_clean_Fin.csv"
OUTPUT_FILE = r"C:\Users\tavou\Downloads\Compressed\top-1m\domains_active_checked.csv"

DOMAIN_COLUMN = "Domain"

BATCH_SIZE = 5000          # تعداد دامنه در هر بچ
DNS_CONCURRENCY = 100      # تعداد همزمانی DNS
DNS_TIMEOUT = 6.0          # timeout هر کوئری DNS (ثانیه)

RESUME = True              # ادامه از خروجی موجود
SAVE_EVERY_BATCH = True    # ذخیره بعد از هر بچ

# DNS Resolver های پایدار
NAMESERVERS = ["8.8.8.8", "1.1.1.1"]


# =========================
# ابزارها
# =========================
def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def dns_result_has_answer(ans) -> bool:
    """
    سازگار با نسخه‌های مختلف aiodns:
    - در نسخه‌های قدیمی ممکن است لیست/iterable برگردد
    - در نسخه جدید query_dns یک DNSResult برمی‌گرداند
      که پاسخ ممکن است در addresses یا hosts باشد.
    """
    if ans is None:
        return False

    # نسخه جدید
    addresses = getattr(ans, "addresses", None)
    if addresses:
        return True

    hosts = getattr(ans, "hosts", None)
    if hosts:
        return True

    # برخی حالت‌ها: iterable/لیست
    try:
        return len(ans) > 0
    except Exception:
        pass

    # fallback
    return bool(ans)


async def dns_check_domain(resolver: aiodns.DNSResolver, domain: str, sem: asyncio.Semaphore):
    """
    خروجی:
      is_active: آیا A یا AAAA دارد
      has_a
      has_aaaa
    """
    async with sem:
        has_a = False
        has_aaaa = False

        # A
        try:
            ans_a = await asyncio.wait_for(resolver.query_dns(domain, "A"), timeout=DNS_TIMEOUT)
            has_a = dns_result_has_answer(ans_a)
        except Exception:
            has_a = False

        # AAAA
        try:
            ans_aaaa = await asyncio.wait_for(resolver.query_dns(domain, "AAAA"), timeout=DNS_TIMEOUT)
            has_aaaa = dns_result_has_answer(ans_aaaa)
        except Exception:
            has_aaaa = False

        is_active = has_a or has_aaaa
        return is_active, has_a, has_aaaa


async def process_batch(resolver: aiodns.DNSResolver, domains: list[str]):
    sem = asyncio.Semaphore(DNS_CONCURRENCY)
    tasks = [dns_check_domain(resolver, d, sem) for d in domains]
    return await asyncio.gather(*tasks)


def detect_resume_index(output_file: str) -> int:
    """
    اگر فایل خروجی وجود داشته باشد و RESUME فعال باشد،
    از تعداد سطرهای خروجی به عنوان ایندکس شروع استفاده می‌کنیم.
    """
    if not RESUME:
        return 0
    if not os.path.exists(output_file):
        return 0
    try:
        out_df = pd.read_csv(output_file)
        return len(out_df)
    except Exception:
        return 0


def validate_input_columns(df: pd.DataFrame):
    required = {"Rank", "Domain", "character_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ستون‌های ضروری در فایل ورودی موجود نیست: {missing}\n"
            f"ستون‌های موجود: {list(df.columns)}"
        )


async def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"فایل ورودی پیدا نشد:\n{INPUT_FILE}")

    print("در حال خواندن فایل ورودی ...")
    df = pd.read_csv(INPUT_FILE)

    validate_input_columns(df)

    # پاک‌سازی اولیه دامنه‌ها
    df[DOMAIN_COLUMN] = df[DOMAIN_COLUMN].astype(str).str.strip()
    df = df[df[DOMAIN_COLUMN].notna() & (df[DOMAIN_COLUMN] != "")]
    df = df.reset_index(drop=True)

    total = len(df)
    if total == 0:
        print("هیچ دامنه‌ای برای بررسی وجود ندارد.")
        return

    start_index = detect_resume_index(OUTPUT_FILE)
    if start_index >= total:
        print("همه رکوردها قبلاً پردازش شده‌اند.")
        return

    print(f"تعداد کل دامنه‌ها: {total:,}")
    print(f"شروع پردازش از ایندکس: {start_index:,}")

    resolver = aiodns.DNSResolver(
        nameservers=NAMESERVERS,
        timeout=DNS_TIMEOUT,
        tries=2
    )

    # اگر از وسط ادامه می‌دهیم، حالت append
    write_header = True
    if start_index > 0 and output_path.exists():
        write_header = False

    processed = start_index
    t0 = time.time()

    # پردازش بچ‌ها
    for batch_start in range(start_index, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        chunk = df.iloc[batch_start:batch_end].copy()

        domains = chunk[DOMAIN_COLUMN].tolist()
        results = await process_batch(resolver, domains)

        chunk["is_active"] = [r[0] for r in results]

        # خروجی 4 ستونه
        out_chunk = chunk[["Rank", "Domain", "character_count", "is_active"]]

        if SAVE_EVERY_BATCH:
            mode = "a" if (output_path.exists() and not write_header) else "w"
            out_chunk.to_csv(
                OUTPUT_FILE,
                index=False,
                mode=mode,
                header=write_header,
                encoding="utf-8-sig"
            )
            write_header = False

        processed = batch_end
        elapsed = time.time() - t0
        done = processed - start_index
        speed = done / elapsed if elapsed > 0 else 0.0
        remaining = total - processed
        eta = remaining / speed if speed > 0 else float("inf")

        active_count = int(out_chunk["is_active"].sum())
        print(
            f"[{processed:,}/{total:,}] "
            f"Batch: {batch_start:,}-{batch_end-1:,} | "
            f"Active in batch: {active_count:,}/{len(out_chunk):,} | "
            f"Speed: {speed:,.1f} domain/s | ETA: {format_eta(eta)}"
        )

    print("\n✅ پردازش کامل شد.")
    print(f"📁 خروجی: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ اجرای برنامه توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\n❌ خطا: {e}")
