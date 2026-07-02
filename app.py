"""
Image Processor v1.0
==============================================================

Standalone Streamlit εργαλείο που:
1. Διαβάζει barcode + rough_desc από Google Sheet
2. Ψάχνει εικόνα με Serper (πολλαπλά queries)
3. Vision verification (Gemini) — filter watermarks, wrong products, promo combos
4. Αν η εικόνα έχει λευκό φόντο → επεξεργάζεται:
   - Auto-crop το προϊόν (remove περιττά κενά)
   - Center σε 1280x720 canvas με λευκό φόντο
5. Αν δεν βρεθεί κατάλληλη εικόνα → skip και προχωράει στην επόμενη
6. Γράφει στο sheet: στήλη L (local filename) + στήλη AB (status)
7. Στο τέλος: ZIP με όλες τις processed εικόνες → download button
"""

import streamlit as st
import gspread
import requests
import json
import time
import re
import base64
import unicodedata
import io
import os
import zipfile
from urllib.parse import urlparse
from datetime import datetime
from PIL import Image, ImageOps

# ==========================================
# ΡΥΘΜΙΣΕΙΣ
# ==========================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
# v1.1: SHEET_URL από secrets ώστε να μη γίνεται δημόσιο σε public repo
SHEET_URL = st.secrets.get("SHEET_URL", "")

GEMINI_VISION_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash:generateContent?key={key}"
)

# Target output
OUTPUT_W = 1280
OUTPUT_H = 720
MARGIN_RATIO = 0.08  # 8% padding γύρω από το προϊόν
MIN_INPUT_SIZE = 300  # ελάχιστη ανάλυση εισόδου (κοντύτερη πλευρά)

# Vision thresholds
VISION_SCORE_MIN = 55  # αρκετά αυστηρό για να φιλτράρει watermarks/wrong products

# Search
MAX_IMAGE_CANDIDATES = 12  # παίρνουμε περισσότερες candidates γιατί φιλτράρουμε αυστηρά
MAX_VISION_CHECKS = 6

# ==========================================
# HELPERS
# ==========================================
def remove_tones(text):
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def domain_of(url):
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def looks_like_promo(title):
    if not title:
        return False
    tokens = ["promo", "duo", "trio", "combo", "set ", "kit ", "pack ",
              "+ ", " + ", "δώρο", "δωρο", "πακέτο", "πακετο", "bundle", "gift"]
    return any(tok in title.lower() for tok in tokens)


def url_is_direct_image(url):
    if not url:
        return False
    u = url.lower().split("?")[0]
    non_image_hints = ("/product/", "/p/", ".html", ".php", "?utm", "/category/")
    if any(h in u for h in non_image_hints):
        return False
    return u.endswith((".jpg", ".jpeg", ".png", ".webp"))


def extract_brand_heuristic(description):
    """Simple brand extraction — πρώτη λέξη αν είναι capitalized."""
    if not description:
        return ""
    words = description.strip().split()
    if not words:
        return ""
    first = words[0]
    if first[0].isupper() or not first[0].isalpha():
        # Greek two-word brand prefixes
        two_word = {"be", "dr", "dr.", "st", "st.", "του", "της", "the"}
        if first.lower().rstrip(".,;:") in two_word and len(words) >= 2:
            return f"{first} {words[1]}"
        return first
    return ""


# ==========================================
# SERPER
# ==========================================
def serper_image_search(query, gl="gr", num=10):
    url = "https://google.serper.dev/images"
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        r = requests.post(url, json={"q": query, "gl": gl, "num": num},
                          headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json().get("images", [])
    except Exception as e:
        st.session_state.setdefault("_errors", []).append(f"Serper: {e}")
    return []


def collect_image_candidates(barcode, rough_desc):
    """
    Επιστρέφει list of {url, title, source} candidates.
    Χρησιμοποιεί πολλαπλά queries με προτίμηση σε manufacturer/pharmacy sites.
    """
    barcode = (barcode or "").strip()
    clean_bc = barcode.lstrip('0') or barcode
    brand = extract_brand_heuristic(rough_desc)
    rough_desc = " ".join((rough_desc or "").split())

    queries = []
    # Q1: brand + rest of description
    if brand:
        rest = rough_desc[len(brand):].strip()
        if rest:
            queries.append(f'"{brand}" {rest}')
        queries.append(f'"{brand}" {clean_bc}')
    # Q2: rough_desc + barcode
    queries.append(f'{rough_desc} {clean_bc}')
    queries.append(rough_desc)

    seen = set()
    candidates = []
    for q in queries:
        results = serper_image_search(q)
        for img in results:
            url = img.get("imageUrl", "")
            if not url or url in seen:
                continue
            seen.add(url)
            title = img.get("title", "")
            if looks_like_promo(title):
                continue
            if not url_is_direct_image(url):
                continue
            candidates.append({
                "url": url,
                "title": title[:80],
                "source": domain_of(img.get("link", "") or img.get("source", "")),
                "query": q[:60],
            })
            if len(candidates) >= MAX_IMAGE_CANDIDATES:
                return candidates
    return candidates


# ==========================================
# IMAGE FETCH & VISION
# ==========================================
def fetch_image(url, timeout=12, max_bytes=6_000_000):
    """Κατεβάζει την εικόνα και επιστρέφει PIL Image + raw bytes."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None, None
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
        if not ctype.startswith("image/"):
            return None, None
        content = r.content[:max_bytes]
        img = Image.open(io.BytesIO(content))
        img.load()
        return img, content
    except Exception:
        return None, None


def vision_check(image_bytes, mime, barcode, rough_desc):
    """
    Ελέγχει την εικόνα με Gemini Vision — επιστρέφει dict:
      {score, matches_product, has_watermark, white_background_dominant, is_promo, reason}
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = f"""Evaluate this candidate product image for a catalog.

PRODUCT:
- Barcode: {barcode}
- Description: {rough_desc}

Return STRICT JSON:
{{
  "matches_product": true|false,
  "has_watermark": true|false,
  "white_background_dominant": true|false,
  "is_promo_or_bundle": true|false,
  "is_front_side": true|false,
  "image_quality": "high"|"medium"|"low",
  "score": 0-100,
  "reason": "brief"
}}

CRITICAL RULES:
- has_watermark = true if ANY store logo, "sample", URL, or credits are visible → score ≤ 20
- is_promo_or_bundle = true if multiple products / gift bundle → score ≤ 25
- matches_product = false if wrong product → score ≤ 15
- is_front_side = false if back/ingredients visible → score ≤ 40
- image_quality = "low" (blurry/pixelated) → score ≤ 40
- white_background_dominant: true means background is mostly white/very light gray/clean
  (NOT lifestyle scenes, colored backdrops, gradients)
- Score 90-100 only if: matches_product=true, no watermark, no promo, front side,
  white background, high quality.
"""
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    try:
        res = requests.post(
            GEMINI_VISION_URL.format(key=GEMINI_API_KEY),
            json=payload, timeout=30,
        )
        # Retry 429/503
        for delay in [8, 20]:
            if res.status_code not in (429, 503):
                break
            time.sleep(delay)
            res = requests.post(
                GEMINI_VISION_URL.format(key=GEMINI_API_KEY),
                json=payload, timeout=30,
            )
        if res.status_code == 200:
            return json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        st.session_state.setdefault("_errors", []).append(f"Vision: {e}")
    return {"score": 0, "reason": "vision failed"}


# ==========================================
# IMAGE PROCESSING (Pillow)
# ==========================================
def autocrop_to_content(img, bg_threshold=245):
    """
    Auto-crop τα λευκά κενά γύρω από το προϊόν.
    bg_threshold: pixels με RGB values ≥ αυτό θεωρούνται background.
    """
    # Convert σε RGB αν είναι RGBA / paletted
    if img.mode != "RGB":
        if img.mode == "RGBA":
            # Paste σε λευκό background
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])  # use alpha as mask
            img = bg
        else:
            img = img.convert("RGB")

    # Grayscale για detection
    gray = img.convert("L")
    # Invert: content becomes bright, background becomes dark
    inv = ImageOps.invert(gray)
    # Threshold: everything above (255 - bg_threshold) counts as content
    bbox = inv.point(lambda p: 255 if p > (255 - bg_threshold) else 0).getbbox()
    if bbox is None:
        return img  # δεν βρήκε content, επιστρέφει original
    # Add τα margin
    return img.crop(bbox)


def fit_to_canvas(img, canvas_w=OUTPUT_W, canvas_h=OUTPUT_H, margin_ratio=MARGIN_RATIO):
    """
    Τοποθετεί το image σε canvas canvas_w × canvas_h με λευκό background,
    κεντραρισμένο, με margin_ratio padding γύρω.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Target size μέσα στο canvas (με margins)
    inner_w = int(canvas_w * (1 - 2 * margin_ratio))
    inner_h = int(canvas_h * (1 - 2 * margin_ratio))

    # Resize διατηρώντας aspect ratio
    img_w, img_h = img.size
    ratio_w = inner_w / img_w
    ratio_h = inner_h / img_h
    ratio = min(ratio_w, ratio_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Κέντραρα σε λευκό canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    paste_x = (canvas_w - new_w) // 2
    paste_y = (canvas_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


def process_image(img):
    """Full pipeline: autocrop → fit to 1280x720 white canvas."""
    cropped = autocrop_to_content(img)
    return fit_to_canvas(cropped)


# ==========================================
# MAIN PIPELINE (per row)
# ==========================================
def find_and_process(barcode, rough_desc, debug_log):
    """
    Βρίσκει τη σωστή εικόνα και την επεξεργάζεται.
    Επιστρέφει (processed_PIL_or_None, status_msg, chosen_url_or_None, chosen_score).
    """
    candidates = collect_image_candidates(barcode, rough_desc)
    debug_log.append(f"  Found {len(candidates)} candidates from search")

    if not candidates:
        return None, "❌ Δεν βρέθηκαν candidates", None, 0

    checked = 0
    for c in candidates:
        if checked >= MAX_VISION_CHECKS:
            debug_log.append(f"  Reached max checks ({MAX_VISION_CHECKS}), stopping")
            break

        # Fetch
        img, raw_bytes = fetch_image(c["url"])
        if img is None:
            debug_log.append(f"  ✗ Fetch failed: {c['url'][:80]}")
            continue

        # Size check
        if min(img.size) < MIN_INPUT_SIZE:
            debug_log.append(f"  ✗ Too small ({img.size[0]}x{img.size[1]}): {c['url'][:80]}")
            continue

        # Vision check
        mime = f"image/{img.format.lower()}" if img.format else "image/jpeg"
        v = vision_check(raw_bytes, mime, barcode, rough_desc)
        score = int(v.get("score", 0) or 0)
        checked += 1

        watermark = v.get("has_watermark", False)
        promo = v.get("is_promo_or_bundle", False)
        matches = v.get("matches_product", False)
        white_bg = v.get("white_background_dominant", False)

        debug_log.append(
            f"  #{checked} score={score} match={matches} watermark={watermark} "
            f"promo={promo} whitebg={white_bg} ({c['source']})"
        )

        # Hard rejects
        if watermark:
            debug_log.append(f"    → rejected: watermark")
            continue
        if promo:
            debug_log.append(f"    → rejected: promo/bundle")
            continue
        if not matches:
            debug_log.append(f"    → rejected: doesn't match product")
            continue
        if not white_bg:
            debug_log.append(f"    → rejected: not white background")
            continue
        if score < VISION_SCORE_MIN:
            debug_log.append(f"    → rejected: score {score} < {VISION_SCORE_MIN}")
            continue

        # Accepted! Process it
        debug_log.append(f"  ✓ ACCEPTED: score={score}, {c['url'][:80]}")
        try:
            processed = process_image(img)
            return processed, f"✅ Processed (score {score})", c["url"], score
        except Exception as e:
            debug_log.append(f"    → processing failed: {e}")
            continue

    return None, f"⚠️ Δεν βρέθηκε αποδεκτή εικόνα ({checked} candidates checked)", None, 0


# ==========================================
# GOOGLE SHEETS
# ==========================================
def load_sheet_data(sh):
    sheet = sh.sheet1
    return sheet, sheet.get_all_values()


def update_row(sheet, row_num, local_filename, status):
    """
    Ενημερώνει:
    - L (index 11, 1-indexed = column 12): local filename
    - AB (index 27, 1-indexed = column 28): status
    """
    sheet.update(range_name=f"L{row_num}", values=[[local_filename]])
    sheet.update(range_name=f"AB{row_num}", values=[[status]])


# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="Image Processor v1.0", page_icon="🖼️")

st.title("🖼️ Image Processor v1.0")
st.caption("Βρίσκει, επαληθεύει και επεξεργάζεται εικόνες σε 1280×720 λευκό φόντο")

st.markdown("---")

col1, col2 = st.columns(2)
start_row = col1.number_input("Από γραμμή:", min_value=2, value=2)
end_row = col2.number_input("Έως γραμμή:", min_value=2, value=10)

skip_processed = st.checkbox(
    "🔄 Skip γραμμές με status='✅ Processed' (για re-runs)",
    value=True,
)

with st.expander("ℹ️ Πληροφορίες"):
    st.markdown(f"""
**Πώς δουλεύει:**
1. Διαβάζει barcode + rough_desc από στήλες D & E
2. Ψάχνει εικόνες με Serper (πολλαπλά queries)
3. Vision check (Gemini) — απορρίπτει watermarks, wrong products, colored backgrounds
4. Αν βρεθεί καλή εικόνα → auto-crop + center σε 1280×720 canvas με λευκό φόντο
5. Save τοπικά, γράφει status στη στήλη AB
6. Στο τέλος: ZIP download με όλες τις εικόνες

**Filters:**
- Vision score min: **{VISION_SCORE_MIN}**
- Min input resolution: **{MIN_INPUT_SIZE}px** στην κοντύτερη πλευρά
- Max candidates check per row: **{MAX_VISION_CHECKS}**
""")

if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = {}  # barcode → PIL image bytes

if st.button("🚀 Start Processing", type="primary"):
    try:
        creds = json.loads(st.secrets["GOOGLE_CREDENTIALS"], strict=False)
        if "private_key" in creds:
            creds["private_key"] = creds["private_key"].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_url(SHEET_URL)
        sheet, data = load_sheet_data(sh)

        # Ensure column AB exists (status)
        # (Sheets auto-expands, δε χρειάζεται τίποτα)

        st.info(f"Ξεκίνησε επεξεργασία γραμμών {start_row}-{end_row}")

        progress_bar = st.progress(0)
        status_box = st.empty()
        stats = {"processed": 0, "not_found": 0, "skipped": 0, "error": 0}
        all_debug = []
        run_start = time.time()

        # Clear previous session images
        st.session_state["processed_images"] = {}

        total = end_row - start_row + 1
        for i, row_num in enumerate(range(start_row, end_row + 1)):
            actual_idx = row_num - 1
            if actual_idx >= len(data):
                stats["skipped"] += 1
                continue

            row = data[actual_idx]
            barcode = row[3].strip() if len(row) > 3 else ""
            rough_desc = row[4].strip() if len(row) > 4 else ""
            existing_status = row[27].strip() if len(row) > 27 else ""

            if not barcode:
                stats["skipped"] += 1
                all_debug.append(f"Γραμμή {row_num}: skipped (empty barcode)")
                continue

            if skip_processed and "✅ Processed" in existing_status:
                stats["skipped"] += 1
                all_debug.append(f"Γραμμή {row_num}: skipped (already processed)")
                continue

            status_box.info(f"⏳ Γραμμή {row_num}: {barcode} — {rough_desc[:50]}")

            row_debug = [f"Γραμμή {row_num} ({barcode}): {rough_desc[:60]}"]
            try:
                processed, status_msg, chosen_url, chosen_score = find_and_process(
                    barcode, rough_desc, row_debug
                )
            except Exception as e:
                stats["error"] += 1
                all_debug.append(f"Γραμμή {row_num}: EXCEPTION {e}")
                progress_bar.progress((i + 1) / total)
                continue

            all_debug.extend(row_debug)

            if processed:
                # Save to session
                buf = io.BytesIO()
                processed.save(buf, format="JPEG", quality=92)
                filename = f"{barcode}.jpg"
                st.session_state["processed_images"][filename] = buf.getvalue()

                # Update sheet
                try:
                    update_row(sheet, row_num, filename, status_msg)
                except Exception as e:
                    all_debug.append(f"  ⚠ sheet update failed: {e}")

                stats["processed"] += 1
                status_box.success(f"✅ Γραμμή {row_num}: {status_msg}")
            else:
                # Update sheet with status only
                try:
                    update_row(sheet, row_num, "", status_msg)
                except Exception:
                    pass
                stats["not_found"] += 1
                status_box.warning(f"⚠️ Γραμμή {row_num}: {status_msg}")

            progress_bar.progress((i + 1) / total)
            time.sleep(1.5)  # Rate limiting για Gemini

        elapsed = time.time() - run_start
        st.balloons()
        st.success(f"🎉 Ολοκληρώθηκε σε {elapsed/60:.1f} λεπτά")

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Processed", stats["processed"])
        c2.metric("⚠️ Not found", stats["not_found"])
        c3.metric("⏭️ Skipped", stats["skipped"])
        c4.metric("❌ Errors", stats["error"])

        # Show images
        if st.session_state["processed_images"]:
            st.markdown("### 🖼️ Preview")
            cols = st.columns(4)
            for idx, (fname, img_bytes) in enumerate(list(st.session_state["processed_images"].items())[:12]):
                cols[idx % 4].image(img_bytes, caption=fname, use_container_width=True)
            if len(st.session_state["processed_images"]) > 12:
                st.caption(f"... και άλλες {len(st.session_state['processed_images']) - 12}")

        # Debug log
        with st.expander(f"🔍 Debug log ({len(all_debug)} lines)"):
            for line in all_debug:
                st.text(line)

    except Exception as e:
        st.error(f"❌ Σφάλμα: {e}")
        st.exception(e)

# ZIP Download button (persistent)
if st.session_state.get("processed_images"):
    st.markdown("---")
    st.markdown("### 📦 Bulk Download")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, img_bytes in st.session_state["processed_images"].items():
            zf.writestr(fname, img_bytes)
    zip_buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=f"⬇️ Download ZIP ({len(st.session_state['processed_images'])} images)",
        data=zip_buf.getvalue(),
        file_name=f"processed_images_{timestamp}.zip",
        mime="application/zip",
    )
