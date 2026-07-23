import csv
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev, pvariance

# =========================
# تنظیمات
# =========================
INPUT_CSV = r"domains_com_only_with_length_stats.csv"
OUTPUT_DIR = r"microgpt_prep"

OUTPUT_FILTERED_CSV = "microgpt_domains_filtered.csv"
OUTPUT_INPUT_TXT = "input.txt"
OUTPUT_REPORT_TXT = "microgpt_data_report.txt"
OUTPUT_STATS_JSON = "microgpt_data_stats.json"

MIN_LEN = 4
MAX_LEN = 20
ALLOWED_PATTERN = re.compile(r"^[a-z]+$")


def extract_domain_name(row):
    """
    اولویت:
    1) domain_name
    2) Domain با حذف .com
    """
    if "domain_name" in row and row["domain_name"]:
        return row["domain_name"].strip().lower()

    if "Domain" in row and row["Domain"]:
        domain = row["Domain"].strip().lower()
        if domain.endswith(".com"):
            return domain[:-4]
        if "." in domain:
            return domain.rsplit(".", 1)[0]
        return domain

    return ""


def is_valid_domain_name(name):
    if not name:
        return False
    if not (MIN_LEN <= len(name) <= MAX_LEN):
        return False
    if not ALLOWED_PATTERN.fullmatch(name):
        return False
    return True


def load_and_filter_rows(input_csv):
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"فایل ورودی پیدا نشد: {input_csv}")

    kept = []
    seen = set()

    total_rows = 0
    missing_name = 0
    invalid_length = 0
    invalid_charset = 0
    duplicates = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("فایل CSV فاقد header است.")

        fieldnames = set(reader.fieldnames)
        if "domain_name" not in fieldnames and "Domain" not in fieldnames:
            raise ValueError(
                f"ستون لازم پیدا نشد. ستون‌های موجود: {reader.fieldnames}\n"
                f"باید یکی از این‌ها موجود باشد: domain_name یا Domain"
            )

        for row in reader:
            total_rows += 1
            name = extract_domain_name(row)

            if not name:
                missing_name += 1
                continue

            if not (MIN_LEN <= len(name) <= MAX_LEN):
                invalid_length += 1
                continue

            if not ALLOWED_PATTERN.fullmatch(name):
                invalid_charset += 1
                continue

            if name in seen:
                duplicates += 1
                continue

            seen.add(name)

            kept.append({
                "domain_name": name,
                "character_count": len(name),
            })

    counters = {
        "total_rows": total_rows,
        "missing_name": missing_name,
        "invalid_length": invalid_length,
        "invalid_charset": invalid_charset,
        "duplicates_removed": duplicates,
        "kept_rows": len(kept),
    }
    return kept, counters


def build_length_frequency(rows):
    freq = {}
    for row in rows:
        cc = row["character_count"]
        freq[cc] = freq.get(cc, 0) + 1

    for row in rows:
        row["length_frequency"] = freq[row["character_count"]]

    return freq


def write_filtered_csv(rows, output_path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain_name", "character_count", "length_frequency"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_input_txt(rows, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["domain_name"] + "\n")


def compute_stats(rows, freq):
    lengths = [row["character_count"] for row in rows]
    if not lengths:
        raise ValueError("پس از فیلتر هیچ دامنه‌ای باقی نماند.")

    stats = {
        "count": len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": mean(lengths),
        "median_length": median(lengths),
        "variance_population": pvariance(lengths) if len(lengths) > 1 else 0.0,
        "std_population": pstdev(lengths) if len(lengths) > 1 else 0.0,
        "length_distribution": dict(sorted(freq.items())),
        "microgpt_requirements": {
            "input_format": "one domain per line in input.txt",
            "allowed_charset": "a-z only",
            "recommended_block_size": 32,
            "minimum_block_size_for_max_len_22": 23,
            "reason": "tokens are [BOS] + chars + [BOS]"
        }
    }
    return stats


def write_report(stats, counters, report_path):
    lines = []
    lines.append("microGPT Training Data Preparation Report")
    lines.append("=" * 48)
    lines.append("")
    lines.append("Filtering policy:")
    lines.append(f"- character_count بین {MIN_LEN} و {MAX_LEN}")
    lines.append("- فقط a-z")
    lines.append("- بدون عدد")
    lines.append("- بدون -")
    lines.append("- حذف موارد تکراری")
    lines.append("")
    lines.append("Row processing summary:")
    lines.append(f"- total_rows: {counters['total_rows']:,}")
    lines.append(f"- missing_name: {counters['missing_name']:,}")
    lines.append(f"- invalid_length: {counters['invalid_length']:,}")
    lines.append(f"- invalid_charset: {counters['invalid_charset']:,}")
    lines.append(f"- duplicates_removed: {counters['duplicates_removed']:,}")
    lines.append(f"- kept_rows: {counters['kept_rows']:,}")
    lines.append("")
    lines.append("Final dataset statistics:")
    lines.append(f"- count: {stats['count']:,}")
    lines.append(f"- min_length: {stats['min_length']}")
    lines.append(f"- max_length: {stats['max_length']}")
    lines.append(f"- mean_length: {stats['mean_length']:.4f}")
    lines.append(f"- median_length: {stats['median_length']:.4f}")
    lines.append(f"- variance_population: {stats['variance_population']:.4f}")
    lines.append(f"- std_population: {stats['std_population']:.4f}")
    lines.append("")
    lines.append("Length distribution:")
    for cc, cnt in stats["length_distribution"].items():
        lines.append(f"- {cc}: {cnt:,}")
    lines.append("")
    lines.append("microGPT requirements:")
    lines.append(f"- input_format: {stats['microgpt_requirements']['input_format']}")
    lines.append(f"- allowed_charset: {stats['microgpt_requirements']['allowed_charset']}")
    lines.append(
        f"- minimum_block_size_for_max_len_22: "
        f"{stats['microgpt_requirements']['minimum_block_size_for_max_len_22']}"
    )
    lines.append(
        f"- recommended_block_size: "
        f"{stats['microgpt_requirements']['recommended_block_size']}"
    )
    lines.append(f"- reason: {stats['microgpt_requirements']['reason']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_stats_json(stats, counters, output_path):
    payload = {
        "filtering_summary": counters,
        "dataset_statistics": stats
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_csv_path = output_dir / OUTPUT_FILTERED_CSV
    input_txt_path = output_dir / OUTPUT_INPUT_TXT
    report_txt_path = output_dir / OUTPUT_REPORT_TXT
    stats_json_path = output_dir / OUTPUT_STATS_JSON

    print("Reading and filtering input CSV ...")
    rows, counters = load_and_filter_rows(INPUT_CSV)

    print("Building length frequency ...")
    freq = build_length_frequency(rows)

    print("Writing filtered CSV ...")
    write_filtered_csv(rows, filtered_csv_path)

    print("Writing input.txt for microGPT ...")
    write_input_txt(rows, input_txt_path)

    print("Computing statistics ...")
    stats = compute_stats(rows, freq)

    print("Writing report files ...")
    write_report(stats, counters, report_txt_path)
    write_stats_json(stats, counters, stats_json_path)

    print("\nDone.")
    print(f"Filtered CSV: {filtered_csv_path}")
    print(f"microGPT input: {input_txt_path}")
    print(f"Report TXT: {report_txt_path}")
    print(f"Stats JSON: {stats_json_path}")
    print(f"Final rows kept: {stats['count']:,}")
    print(f"Mean length: {stats['mean_length']:.4f}")
    print(f"Median length: {stats['median_length']:.4f}")
    print(f"Recommended microGPT block_size: {stats['microgpt_requirements']['recommended_block_size']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
