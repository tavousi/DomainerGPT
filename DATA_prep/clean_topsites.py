import pandas as pd
import re

# -----------------------------
# تنظیم نام فایل‌ها
# -----------------------------
INPUT_FILE = "tranco_64YYX.csv"
OUTPUT_CLEAN = "domains_clean_Fin.csv"
OUTPUT_REMOVED = "domains_removed_Fin.csv"

# -----------------------------
# خواندن فایل
# -----------------------------
df = pd.read_csv(
    INPUT_FILE,
    header=None,
    names=["Rank", "Domain"]
)

# حذف فاصله‌های ابتدا و انتها
df["Domain"] = df["Domain"].str.strip()

# -----------------------------
# شمارش کاراکترها
# (نام کامل دامنه شامل پسوند)
# -----------------------------
df["character_count"] = df["Domain"].str.len()

# -----------------------------
# الگوی مجاز
#
# فقط:
# a-z
# A-Z
# 0-9
# .
# -
#
# شرایط:
# - با حرف یا عدد شروع شود.
# - با حرف یا عدد پایان یابد.
# - هیچ برچسبی (label) با - شروع یا تمام نشود.
# -----------------------------
pattern = re.compile(
    r'^(?!-)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$'
)

# -----------------------------
# تابع اعتبارسنجی
# -----------------------------
def is_valid(domain):

    if not (6 <= len(domain) <= 66):
        return False

    if pattern.fullmatch(domain) is None:
        return False

    return True

# -----------------------------
# اعمال فیلتر
# -----------------------------
mask = df["Domain"].apply(is_valid)

clean = df[mask].copy()
removed = df[~mask].copy()

# -----------------------------
# فقط ستون‌های مورد نیاز
# -----------------------------
clean = clean[["Rank", "Domain", "character_count"]]
removed = removed[["Rank", "Domain", "character_count"]]

# -----------------------------
# ذخیره
# -----------------------------
clean.to_csv(
    OUTPUT_CLEAN,
    index=False,
    encoding="utf-8-sig"
)

removed.to_csv(
    OUTPUT_REMOVED,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print("Finished")
print(f"Valid domains   : {len(clean):,}")
print(f"Removed domains : {len(removed):,}")
print(f"Saved -> {OUTPUT_CLEAN}")
print(f"Saved -> {OUTPUT_REMOVED}")
