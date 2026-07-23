import pandas as pd
from pathlib import Path

# =========================
# تنظیمات
# =========================
INPUT_FILE = r"C:\Users\tavou\Downloads\Compressed\top-1m\domains_com_only.csv"
OUTPUT_DATA_FILE = r"C:\Users\tavou\Downloads\Compressed\top-1m\domains_com_only_with_length_stats.csv"
OUTPUT_FREQ_FILE = r"C:\Users\tavou\Downloads\Compressed\top-1m\domains_com_length_frequency.csv"
OUTPUT_REPORT_FILE = r"C:\Users\tavou\Downloads\Compressed\top-1m\domains_com_statistical_report.txt"


def extract_domain_name_from_com(domain: str) -> str:
    """
    example.com -> example
    mail.example.com -> mail.example
    """
    if not isinstance(domain, str):
        return ""

    domain = domain.strip().lower()

    if domain.endswith(".com"):
        return domain[:-4]

    if "." in domain:
        return domain.rsplit(".", 1)[0]

    return domain


def build_frequency_table(series: pd.Series) -> pd.DataFrame:
    freq = series.value_counts().sort_index()
    freq_df = freq.reset_index()
    freq_df.columns = ["character_count", "domain_count"]
    return freq_df


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"فایل ورودی پیدا نشد:\n{INPUT_FILE}")

    print("در حال خواندن فایل ...")
    df = pd.read_csv(INPUT_FILE)

    if "domain_name" in df.columns:
        df["domain_name"] = df["domain_name"].astype(str).str.strip().str.lower()
    elif "Domain" in df.columns:
        df["Domain"] = df["Domain"].astype(str).str.strip().str.lower()
        df["domain_name"] = df["Domain"].apply(extract_domain_name_from_com)
    else:
        raise ValueError(
            "هیچ‌کدام از ستون‌های 'domain_name' یا 'Domain' در فایل موجود نیست.\n"
            f"ستون‌های موجود: {list(df.columns)}"
        )

    df = df[df["domain_name"].notna()].copy()
    df["domain_name"] = df["domain_name"].astype(str).str.strip()
    df = df[df["domain_name"] != ""].copy()

    # اگر character_count نبود، بساز
    if "character_count" not in df.columns:
        df["character_count"] = df["domain_name"].str.len()
    else:
        df["character_count"] = pd.to_numeric(df["character_count"], errors="coerce")
        missing_mask = df["character_count"].isna()
        df.loc[missing_mask, "character_count"] = df.loc[missing_mask, "domain_name"].str.len()

    df["character_count"] = df["character_count"].astype(int)

    # جدول فراوانی طول‌ها
    freq_df = build_frequency_table(df["character_count"])

    # الصاق تعداد هر طول به تمام ردیف‌های همان طول
    freq_map = dict(zip(freq_df["character_count"], freq_df["domain_count"]))
    df["length_frequency"] = df["character_count"].map(freq_map).astype(int)

    # ذخیره فایل اصلی خروجی
    df.to_csv(OUTPUT_DATA_FILE, index=False, encoding="utf-8-sig")

    # ذخیره جدول فراوانی
    freq_df.to_csv(OUTPUT_FREQ_FILE, index=False, encoding="utf-8-sig")

    # آمار کلی
    total = int(len(df))
    min_len = int(df["character_count"].min())
    max_len = int(df["character_count"].max())
    mean_len = float(df["character_count"].mean())
    median_len = float(df["character_count"].median())
    variance_len = float(df["character_count"].var(ddof=1)) if total > 1 else 0.0
    std_len = float(df["character_count"].std(ddof=1)) if total > 1 else 0.0
    mode_series = df["character_count"].mode()
    modes = ", ".join(str(int(x)) for x in mode_series.tolist()) if not mode_series.empty else ""

    # بازه‌های رایج‌تر برای تصمیم‌گیری
    q1 = float(df["character_count"].quantile(0.25))
    q3 = float(df["character_count"].quantile(0.75))
    p10 = float(df["character_count"].quantile(0.10))
    p90 = float(df["character_count"].quantile(0.90))
    p95 = float(df["character_count"].quantile(0.95))

    lines = []
    lines.append("گزارش آماری طول نام دامنه‌های .com")
    lines.append("=" * 50)
    lines.append(f"تعداد کل دامنه‌ها: {total:,}")
    lines.append(f"حداقل طول: {min_len}")
    lines.append(f"حداکثر طول: {max_len}")
    lines.append(f"میانگین: {mean_len:.4f}")
    lines.append(f"میانه: {median_len:.4f}")
    lines.append(f"واریانس: {variance_len:.4f}")
    lines.append(f"انحراف معیار: {std_len:.4f}")
    lines.append(f"نما (Mode): {modes}")
    lines.append(f"چارک اول Q1: {q1:.4f}")
    lines.append(f"چارک سوم Q3: {q3:.4f}")
    lines.append(f"صدک 10: {p10:.4f}")
    lines.append(f"صدک 90: {p90:.4f}")
    lines.append(f"صدک 95: {p95:.4f}")
    lines.append("")
    lines.append("فراوانی طول‌ها")
    lines.append("-" * 50)

    for _, row in freq_df.iterrows():
        cc = int(row["character_count"])
        cnt = int(row["domain_count"])
        pct = (cnt / total) * 100 if total > 0 else 0
        lines.append(f"character_count = {cc:>2} | count = {cnt:>8,} | percent = {pct:>6.2f}%")

    report_text = "\n".join(lines)
    Path(OUTPUT_REPORT_FILE).write_text(report_text, encoding="utf-8")

    print("پردازش کامل شد.")
    print(f"فایل خروجی اصلی: {OUTPUT_DATA_FILE}")
    print(f"فایل فراوانی طول‌ها: {OUTPUT_FREQ_FILE}")
    print(f"فایل گزارش آماری: {OUTPUT_REPORT_FILE}")
    print(f"تعداد کل دامنه‌ها: {total:,}")
    print(f"حداقل طول: {min_len}")
    print(f"حداکثر طول: {max_len}")
    print(f"میانگین: {mean_len:.4f}")
    print(f"میانه: {median_len:.4f}")
    print(f"انحراف معیار: {std_len:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nاجرا توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\nخطا: {e}")
