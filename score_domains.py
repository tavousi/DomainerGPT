#!/usr/bin/env python3
"""
score_domains.py — Read available domains, score them based on heuristics, and sort them.
"""

import os
import re

INPUT_FILE = "available_domains_com.txt"
OUTPUT_FILE = "premium_domains_com.txt"

# لیست کلمات کلیدی باارزش در ثبت دامنه
PREMIUM_KEYWORDS = [
    'app', 'tech', 'ai', 'pro', 'hub', 'net', 'host', 'bet', 
    'pay', 'coin', 'news', 'shop', 'go', 'web', 'smart', 'cloud', 
    'data', 'sys', 'lab', 'now', 'top', 'buy', 'game'
]

def score_domain(domain):
    """
    محاسبه ارزش تجاری و خوش‌آهنگی دامنه.
    امتیاز پایه ۱۰۰ است و بر اساس ویژگی‌ها کم یا زیاد می‌شود.
    """
    score = 100
    name = domain.split('.')[0].lower()
    
    # ۱. ارزیابی طول دامنه (کوتاه‌تر = بهتر)
    length = len(name)
    if length <= 5:
        score += 30
    elif length == 6:
        score += 15
    elif length > 8:
        score -= (length - 8) * 10  # جریمه برای دامنه‌های طولانی
        
    # ۲. ارزیابی آواشناسی و خوانایی
    vowels = "aeiouy"
    v_count = sum(1 for char in name if char in vowels)
    c_count = length - v_count
    
    # اگر کلاً حرف صدادار یا بی‌صدا ندارد (غیرقابل تلفظ)
    if v_count == 0 or c_count == 0:
        score -= 60
    else:
        # نسبت طلایی حروف صدادار (بین ۳۰ تا ۶۰ درصد)
        ratio = v_count / length
        if 0.3 <= ratio <= 0.6:
            score += 15
            
    # جریمه برای حروف بی‌صدای متوالی (سخت شدن تلفظ)
    if re.search(r'[^aeiouy]{4,}', name):
        score -= 40  # ۴ حرف بی‌صدا پشت سر هم
    elif re.search(r'[^aeiouy]{3,}', name):
        score -= 15  # ۳ حرف بی‌صدا پشت سر هم
        
    # ۳. ارزیابی کلمات کلیدی تجاری
    found_keywords = []
    for kw in PREMIUM_KEYWORDS:
        if kw in name:
            score += 25
            found_keywords.append(kw)
            
    # امتیاز منفی برای شروع یا پایان با کلمات نامناسب (مانند x یا z در جاهای عجیب)
    if name.startswith('x') or name.endswith('x'):
        score -= 10
        
    return score, found_keywords

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"خطا: فایل '{INPUT_FILE}' پیدا نشد.")
        return

    print("در حال تحلیل و امتیازبندی دامنه‌ها...")
    
    scored_domains = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        domains = [line.strip() for line in f if line.strip()]
        
    for dom in domains:
        score, kws = score_domain(dom)
        scored_domains.append((score, dom, kws))
        
    # مرتب‌سازی بر اساس امتیاز (نزولی)
    scored_domains.sort(key=lambda x: x[0], reverse=True)
    
    # ذخیره در فایل جدید
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'Domain':<20} | {'Score':<6} | {'Keywords'}\n")
        f.write("-" * 50 + "\n")
        for score, dom, kws in scored_domains:
            kws_str = ", ".join(kws) if kws else "-"
            f.write(f"{dom:<20} | {score:<6} | {kws_str}\n")
            
    # نمایش ۲۰ دامنه برتر در کنسول
    print("\n" + "★ "*25)
    print(" ۲۰ دامنه برتر (Premium):")
    print("-" * 50)
    print(f"{'دامنه (Domain)':<20} | {'امتیاز':<6} | {'کلمات کلیدی'}")
    print("-" * 50)
    
    for score, dom, kws in scored_domains[:20]:
        kws_str = ", ".join(kws) if kws else "-"
        print(f"{dom:<20} | {score:<6} | {kws_str}")
        
    print("-" * 50)
    print(f"لیست کامل دامنه‌های مرتب‌شده در '{OUTPUT_FILE}' ذخیره شد.")

if __name__ == "__main__":
    main()