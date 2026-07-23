import pandas as pd
from pathlib import Path

# =========================
# تنظیمات فایل‌ها
# =========================
INPUT_FILE = r"domains_active_checked.csv"

OUTPUT_ACTIVE_FILE = r"domains_active_filtered.csv"
OUTPUT_COM_FILE = r"domains_com_only.csv"


def is_punycode_domain(domain: str) -> bool:
    """
    اگر هر بخش از دامنه با xn-- شروع شود، آن را IDN/Punycode در نظر می‌گیریم.
    نمونه:
      xn--abc.com
      test.xn--p1ai
    """
    if not isinstance(domain, str):
        return False
    labels = domain.lower().strip().split(".")
    return any(label.startswith("xn--") for label in labels)


def extract_tld(domain: str) -> str:
    """
    آخرین بخش دامنه را به عنوان TLD برمی‌گرداند.
    google.com -> com
    example.org -> org
    """
    if not isinstance(domain, str):
        return ""
    domain = domain.strip().lower()
    if "." not in domain:
        return ""
    return domain.rsplit(".", 1)[-1]


def extract_domain_name(domain: str) -> str:
    """
    نام دامنه را بدون پسوند نهایی برمی‌گرداند.
    google.com -> google
    mail.google.com -> mail.google
    example.co.uk -> example.co
    """
    if not isinstance(domain, str):
        return ""
    domain = domain.strip().lower()
    if "." not in domain:
        return domain
    return domain.rsplit(".", 1)[0]


def normalize_is_active(value) -> bool:
    """
    سازگار با حالت‌های مختلف ذخیره‌شدن True/False در CSV
    """
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"فایل ورودی پیدا نشد:\n{INPUT_FILE}")

    print("در حال خواندن فایل خروجی قبلی ...")
    df = pd.read_csv(INPUT_FILE)

    required_columns = {"Rank", "Domain", "character_count", "is_active"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"ستون‌های ضروری پیدا نشد: {missing}\n"
            f"ستون‌های موجود: {list(df.columns)}"
        )

    # پاک‌سازی اولیه
    df["Domain"] = df["Domain"].astype(str).str.strip().str.lower()
    df["is_active"] = df["is_active"].apply(normalize_is_active)

    total_before = len(df)

    # فقط دامنه‌های active
    df = df[df["is_active"]].copy()
    after_active = len(df)

    # حذف دامنه‌های punycode / xn--
    df = df[~df["Domain"].apply(is_punycode_domain)].copy()
    after_punycode_removed = len(df)

    # افزودن ستون‌های جدید
    df["IDN_TLD"] = df["Domain"].apply(extract_tld)
    df["domain_name"] = df["Domain"].apply(extract_domain_name)

    # مرتب‌سازی ستون‌ها برای خروجی اول
    output_active = df[
        ["Rank", "Domain", "character_count", "is_active", "IDN_TLD", "domain_name"]
    ].copy()

    output_active.to_csv(OUTPUT_ACTIVE_FILE, index=False, encoding="utf-8-sig")

    # خروجی دوم: فقط .com
    com_df = df[df["IDN_TLD"] == "com"].copy()

    output_com = com_df[
        ["Rank", "Domain", "character_count", "domain_name"]
    ].copy()

    output_com.to_csv(OUTPUT_COM_FILE, index=False, encoding="utf-8-sig")

    print("پردازش کامل شد.")
    print(f"تعداد کل رکوردهای ورودی: {total_before:,}")
    print(f"تعداد رکوردهای active: {after_active:,}")
    print(f"تعداد رکوردها پس از حذف xn--: {after_punycode_removed:,}")
    print(f"تعداد دامنه‌های .com: {len(output_com):,}")
    print(f"خروجی اول: {OUTPUT_ACTIVE_FILE}")
    print(f"خروجی دوم: {OUTPUT_COM_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nاجرا توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\nخطا: {e}")
