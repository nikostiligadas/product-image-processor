"""
Image Processor v1.4 — με Google Drive upload
==============================================================

Standalone Streamlit εργαλείο που:
1. Διαβάζει barcode + rough_desc από Google Sheet
2. Ψάχνει εικόνα με Serper (πολλαπλά queries)
3. Vision verification (Gemini) — filter watermarks, wrong products, promo combos
4. Αν η εικόνα έχει λευκό φόντο → επεξεργάζεται:
   - Auto-crop το προϊόν (remove περιττά κενά)
   - Center σε 1280x720 canvas με λευκό φόντο
5. Αν δεν βρεθεί κατάλληλη εικόνα → skip
6. Ανεβάζει την επεξεργασμένη εικόνα σε Google Drive (public)
7. Γράφει στο sheet: στήλη L (public URL) + στήλη AB (status)

Filename: {barcode}_1280x720.jpg
URL format: https://drive.google.com/uc?export=view&id=FILE_ID
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
from urllib.parse import urlparse
from datetime import datetime
from PIL import Image, ImageOps
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.service_account import Credentials

# ==========================================
# ΡΥΘΜΙΣΕΙΣ
# ==========================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
SHEET_URL = st.secrets.get("SHEET_URL", "")
# v1.1: Google Drive folder ID για το image hosting
DRIVE_FOLDER_ID = st.secrets.get("DRIVE_FOLDER_ID", "")

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
MAX_IMAGE_CANDIDATES = 15  # παίρνουμε περισσότερες candidates γιατί φιλτράρουμε αυστηρά
MAX_VISION_CHECKS = 8  # v1.4: αυξήθηκε από 6 → 8 για δύσκολα cases (π.χ. promo-heavy προϊόντα)

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
    serper_queries_used = 0
    for q in queries:
        results = serper_image_search(q)
        serper_queries_used += 1
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
                return candidates, serper_queries_used
    return candidates, serper_queries_used


# ==========================================
# IMAGE FETCH & VISION
# ==========================================
def fetch_image(url, timeout=12, max_bytes=6_000_000):
    """Κατεβάζει την εικόνα και επιστρέφει PIL Image + raw bytes."""
    # v1.4: Browser-like headers ώστε sites όπως wecare.gr, blinkshop κλπ
    # να μη μπλοκάρουν το request ως bot. Αυτό αυξάνει σημαντικά το
    # success rate των downloads.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "el-GR,el;q=0.9,en;q=0.8",
        "Referer": "https://www.google.com/",
    }
    try:
        r = requests.get(url, timeout=timeout, headers=headers, stream=True)
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
# GOOGLE DRIVE UPLOAD (v1.1)
# ==========================================
_drive_service_cache = None


def get_drive_service():
    """Lazy-load Google Drive API client, cached σε module level."""
    global _drive_service_cache
    if _drive_service_cache is not None:
        return _drive_service_cache
    creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"], strict=False)
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace('\\n', '\n')
    credentials = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    _drive_service_cache = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return _drive_service_cache


def upload_to_drive(image_pil, filename, folder_id):
    """
    Ανεβάζει PIL image σε Google Drive.
    Επιστρέφει public URL ή None σε αποτυχία.
    """
    if not folder_id:
        return None
    try:
        # Save PIL image σε bytes
        buf = io.BytesIO()
        image_pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)

        service = get_drive_service()

        # Check αν υπάρχει ήδη file με ίδιο όνομα στο folder
        # (αν ναι, το κάνουμε overwrite για να μη γεμίσει με duplicates)
        existing = service.files().list(
            q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
            fields="files(id)",
        ).execute()

        media = MediaIoBaseUpload(buf, mimetype="image/jpeg", resumable=False)

        if existing.get("files"):
            # Update existing
            file_id = existing["files"][0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            # Create new
            file_metadata = {
                "name": filename,
                "parents": [folder_id],
            }
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id",
            ).execute()
            file_id = file.get("id")

            # Make it public (Anyone with link → Viewer)
            service.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
                fields="id",
            ).execute()

        # Public URL — direct-view format works για most PIM systems
        return f"https://drive.google.com/uc?export=view&id={file_id}"

    except Exception as e:
        st.session_state.setdefault("_errors", []).append(
            f"Drive upload για {filename} απέτυχε: {e}"
        )
        return None


# ==========================================
# MAIN PIPELINE (per row)
# ==========================================
def find_and_process(barcode, rough_desc, debug_log):
    """
    Βρίσκει τη σωστή εικόνα και την επεξεργάζεται.
    Επιστρέφει (processed_PIL_or_None, status_msg, chosen_url_or_None, chosen_score, usage).
    usage = {"serper": N, "vision": M}
    """
    candidates, serper_used = collect_image_candidates(barcode, rough_desc)
    debug_log.append(f"  Found {len(candidates)} candidates from search ({serper_used} Serper queries)")

    usage = {"serper": serper_used, "vision": 0}

    if not candidates:
        return None, "❌ Δεν βρέθηκαν candidates", None, 0, usage

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
        usage["vision"] += 1

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
            return processed, f"✅ Processed (score {score})", c["url"], score, usage
        except Exception as e:
            debug_log.append(f"    → processing failed: {e}")
            continue

    return None, f"⚠️ Δεν βρέθηκε αποδεκτή εικόνα ({checked} candidates checked)", None, 0, usage


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
st.set_page_config(page_title="Image Processor v1.4", page_icon="🖼️")

st.title("🖼️ Image Processor v1.4")
st.caption("Βρίσκει, επαληθεύει και επεξεργάζεται εικόνες σε 1280×720 λευκό φόντο → Google Drive")

# v1.1: Warning αν λείπει το DRIVE_FOLDER_ID
if not DRIVE_FOLDER_ID:
    st.error(
        "⚠️ Λείπει το `DRIVE_FOLDER_ID` από τα secrets. "
        "Πρόσθεσε στο Streamlit Cloud → Settings → Secrets:\n\n"
        "`DRIVE_FOLDER_ID = \"το_folder_id_σου\"`"
    )
    st.stop()

st.markdown("---")

col1, col2 = st.columns(2)
start_row = col1.number_input("Από γραμμή:", min_value=2, value=2)
end_row = col2.number_input("Έως γραμμή:", min_value=2, value=10)

skip_processed = st.checkbox(
    "🔄 Skip γραμμές με status='✅ Processed' (για re-runs)",
    value=True,
)

with st.expander("ℹ️ Πώς δουλεύει"):
    st.markdown(f"""
**Ροή:**
1. Διαβάζει barcode + rough_desc από στήλες D & E
2. Ψάχνει εικόνες με Serper (πολλαπλά queries)
3. Vision check (Gemini) — απορρίπτει watermarks, wrong products, colored backgrounds
4. Αν βρεθεί καλή εικόνα → auto-crop + center σε 1280×720 canvas με λευκό φόντο
5. **Ανεβάζει την εικόνα σε Google Drive** (public URL)
6. Γράφει στο sheet:
   - Στήλη **L**: το public URL της εικόνας
   - Στήλη **AB**: status (✅ Processed / ⚠️ Not found)

**Filename**: `{{barcode}}_1280x720.jpg`

**Filters:**
- Vision score min: **{VISION_SCORE_MIN}**
- Min input resolution: **{MIN_INPUT_SIZE}px** στην κοντύτερη πλευρά
- Max candidates check per row: **{MAX_VISION_CHECKS}**

**Παράδειγμα URL:** `https://drive.google.com/uc?export=view&id=XXXXXX`
""")

if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = {}  # barcode → PIL image bytes

# v1.2: Stop flag για διακοπή του processing
if "stop_requested" not in st.session_state:
    st.session_state["stop_requested"] = False

col_start, col_stop = st.columns([3, 1])
start_clicked = col_start.button("🚀 Start Processing", type="primary")
if col_stop.button("⏹️ Stop", type="secondary"):
    st.session_state["stop_requested"] = True
    st.warning("⏹️ Ζητήθηκε διακοπή — θα σταματήσει μετά την τρέχουσα γραμμή.")

if start_clicked:
    # Reset stop flag στην αρχή κάθε run
    st.session_state["stop_requested"] = False
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

        # v1.3: Cost & time tracking
        # Pricing references:
        #   Gemini 2.5 Flash Vision: ~$0.075/1M input, $0.30/1M output tokens
        #     ανά vision call: ~1250 in (image + prompt) + 150 out → ~$0.00014
        #   Serper.dev: $0.30 / 1000 queries → $0.0003/query
        # Ανά γραμμή (τυπικά): έως 4 Serper queries + έως 6 vision checks
        cost_gemini = 0.0
        cost_serper = 0.0
        row_times = []  # χρόνος ανά γραμμή για avg

        # Clear previous session images
        st.session_state["processed_images"] = {}

        total = end_row - start_row + 1
        for i, row_num in enumerate(range(start_row, end_row + 1)):
            row_start_time = time.time()  # v1.3: per-row timing
            # v1.2: Stop check
            if st.session_state.get("stop_requested"):
                all_debug.append(
                    f"⏹️ Σταμάτησε στη γραμμή {row_num} κατόπιν αιτήματος χρήστη"
                )
                status_box.warning(
                    f"⏹️ Διακοπή στη γραμμή {row_num}. "
                    f"Όσες γραμμές προηγήθηκαν έχουν γραφτεί στο sheet."
                )
                break
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
                processed, status_msg, chosen_url, chosen_score, usage = find_and_process(
                    barcode, rough_desc, row_debug
                )
            except Exception as e:
                stats["error"] += 1
                all_debug.append(f"Γραμμή {row_num}: EXCEPTION {e}")
                progress_bar.progress((i + 1) / total)
                continue

            all_debug.extend(row_debug)

            # v1.3: Cost accumulation
            cost_serper += usage.get("serper", 0) * 0.0003
            cost_gemini += usage.get("vision", 0) * 0.00014

            if processed:
                # v1.1: Upload σε Google Drive αντί για local session storage
                filename = f"{barcode}_1280x720.jpg"
                drive_url = upload_to_drive(processed, filename, DRIVE_FOLDER_ID)

                # Keep preview στο session (για UI display μόνο)
                buf = io.BytesIO()
                processed.save(buf, format="JPEG", quality=92)
                st.session_state["processed_images"][filename] = buf.getvalue()

                if drive_url:
                    # Update sheet: L = public URL, AB = status
                    try:
                        update_row(sheet, row_num, drive_url, status_msg)
                    except Exception as e:
                        all_debug.append(f"  ⚠ sheet update failed: {e}")
                    stats["processed"] += 1
                    status_box.success(f"✅ Γραμμή {row_num}: {status_msg} → {drive_url[:60]}...")
                else:
                    # Upload failed
                    try:
                        update_row(sheet, row_num, "", "❌ Drive upload failed")
                    except Exception:
                        pass
                    stats["error"] += 1
                    status_box.error(f"❌ Γραμμή {row_num}: Drive upload failed")
            else:
                # Update sheet with status only
                try:
                    update_row(sheet, row_num, "", status_msg)
                except Exception:
                    pass
                stats["not_found"] += 1
                status_box.warning(f"⚠️ Γραμμή {row_num}: {status_msg}")

            progress_bar.progress((i + 1) / total)
            # v1.3: Record row time (πριν το sleep για καθαρό processing time)
            row_times.append(time.time() - row_start_time)
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

        # v1.3: Cost & Time dashboard
        st.markdown("### 💰 Κόστος & Χρόνος")
        cost_total = cost_gemini + cost_serper
        rows_done = stats["processed"] + stats["not_found"] + stats["error"]

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("💵 Total cost", f"${cost_total:.4f}")
        cc1.caption(f"Gemini Vision: ${cost_gemini:.4f} · Serper: ${cost_serper:.4f}")

        if rows_done > 0:
            cost_per_row = cost_total / rows_done
            # Χρησιμοποιούμε το row_times (καθαρός χρόνος χωρίς sleep) για avg
            if row_times:
                avg_time = sum(row_times) / len(row_times)
            else:
                avg_time = elapsed / rows_done
            cc2.metric("📈 Cost/row", f"${cost_per_row:.5f}")
            cc2.caption(f"Επεξεργάστηκαν {rows_done} γραμμές")
            cc3.metric("⏱️ Time/row", f"{avg_time:.1f}s")
            cc3.caption(f"Συνολικός χρόνος: {elapsed/60:.1f} min")

            # Projection για 1000 γραμμές
            proj_cost = cost_per_row * 1000
            # Ρεαλιστικός χρόνος: avg_time + 1.5s sleep ανά γραμμή
            proj_time_hours = ((avg_time + 1.5) * 1000) / 3600
            st.info(
                f"📊 **Projection για 1000 γραμμές με αυτό το mix:** "
                f"~${proj_cost:.2f} · ~{proj_time_hours:.1f} ώρες "
                f"({proj_time_hours*60:.0f} λεπτά)"
            )

            with st.expander("📖 Reference: pricing & calls"):
                st.markdown(f"""
**Pricing reference:**
- Gemini 2.5 Flash Vision: ~$0.00014 ανά image check
- Serper.dev: $0.0003 ανά query

**Calls ανά γραμμή (τυπικά):**
- Serper queries: έως 4 (brand+rest, brand+barcode, desc+barcode, desc)
- Vision checks: έως {MAX_VISION_CHECKS} (σταματάει μόλις βρει αποδεκτή)

**Αυτό το run:**
- Σύνολο Serper queries: {int(cost_serper / 0.0003)}
- Σύνολο Vision checks: {int(cost_gemini / 0.00014)}
- Μέσος χρόνος/γραμμή: {avg_time:.1f}s (καθαρός, χωρίς rate-limit sleep)
""")
        else:
            cc2.metric("📈 Cost/row", "—")
            cc3.metric("⏱️ Time/row", f"{elapsed:.1f}s total")

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

        # v1.1: Errors expander (Drive/Serper/Vision)
        errors = st.session_state.get("_errors", [])
        if errors:
            with st.expander(f"⚠️ Errors ({len(errors)})", expanded=True):
                for err in errors:
                    st.text(err)
            st.session_state["_errors"] = []

    except Exception as e:
        st.error(f"❌ Σφάλμα: {e}")
        st.exception(e)

# v1.1: ZIP download αφαιρέθηκε — τα URLs είναι πλέον στο sheet (στήλη L)
# και οι εικόνες hosted στο Google Drive folder.
