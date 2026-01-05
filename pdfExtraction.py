#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import statistics
from collections import Counter, defaultdict
import fitz  # PyMuPDF


# ----------------------------
# Utilities
# ----------------------------

def norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_page_number(s: str) -> bool:
    s = norm_text(s)
    return bool(re.fullmatch(r"\d{1,4}", s))

def line_bbox(line):
    x0 = min(sp["bbox"][0] for sp in line["spans"])
    y0 = min(sp["bbox"][1] for sp in line["spans"])
    x1 = max(sp["bbox"][2] for sp in line["spans"])
    y1 = max(sp["bbox"][3] for sp in line["spans"])
    return (x0, y0, x1, y1)

def median_font_size(line) -> float:
    sizes = [sp["size"] for sp in line["spans"] if sp.get("text", "").strip()]
    return statistics.median(sizes) if sizes else 0.0

def is_bold(span) -> bool:
    # PyMuPDF may include "Bold" in font name for many PDFs
    fn = (span.get("font") or "").lower()
    return "bold" in fn or "black" in fn

def starts_like_list_item(text: str) -> bool:
    t = text.lstrip()
    return bool(re.match(r"^(\u2022|•|\-|\–|\—)\s+", t) or
                re.match(r"^(\(?\d+\)?[.)])\s+", t) or
                re.match(r"^(\(?[a-zA-Z]\)?[.)])\s+", t))

def strip_list_marker(text: str) -> str:
    t = text.lstrip()
    t = re.sub(r"^(\u2022|•|\-|\–|\—)\s+", "", t)
    t = re.sub(r"^(\(?\d+\)?[.)])\s+", "", t)
    t = re.sub(r"^(\(?[a-zA-Z]\)?[.)])\s+", "", t)
    return t.strip()

def hyphenation_join(prev: str, nxt: str) -> str:
    # Join "inter-" + "national" => "international" when next starts lowercase
    if prev.endswith("-") and nxt and nxt[0].islower():
        return prev[:-1] + nxt
    return prev + " " + nxt


# ----------------------------
# Step 1: Extract lines with geometry
# ----------------------------

def extract_lines(page: fitz.Page):
    """
    Returns a list of line dicts:
    {
      'text': str,
      'bbox': (x0,y0,x1,y1),
      'x0': float, 'y0': float, 'x1': float, 'y1': float,
      'size': float (median),
      'bold': bool,
      'raw': original line object
    }
    """
    td = page.get_text("dict")  # includes blocks/lines/spans with bboxes
    lines_out = []

    for b in td.get("blocks", []):
        if b.get("type") != 0:  # 0=text, 1=image
            continue
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            # Build line text by concatenating spans; keep spacing tight
            parts = []
            for sp in spans:
                tx = sp.get("text", "")
                if tx:
                    parts.append(tx)
            text = norm_text("".join(parts))
            if not text:
                continue

            x0, y0, x1, y1 = line_bbox(ln)
            size = median_font_size(ln)
            bold = any(is_bold(sp) for sp in spans)

            lines_out.append({
                "text": text,
                "bbox": (x0, y0, x1, y1),
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "size": size,
                "bold": bold,
                "raw": ln,
            })
    return lines_out


# ----------------------------
# Step 2: Detect & remove headers/footers
# ----------------------------

def detect_repeated_header_footer(all_pages_lines, page_heights, top_ratio=0.10, bottom_ratio=0.10, freq_threshold=0.6):
    """
    Find texts that repeat across pages within top/bottom bands.
    Returns sets: header_texts, footer_texts
    """
    header_counter = Counter()
    footer_counter = Counter()
    total_pages = len(all_pages_lines)

    for p_idx, lines in enumerate(all_pages_lines):
        H = page_heights[p_idx]
        top_y = H * top_ratio
        bot_y = H * (1 - bottom_ratio)
        for ln in lines:
            t = norm_text(ln["text"])
            if not t:
                continue
            # Ignore pure page numbers in repetition counts
            if is_page_number(t):
                continue
            if ln["y1"] <= top_y:
                header_counter[t] += 1
            elif ln["y0"] >= bot_y:
                footer_counter[t] += 1

    header_texts = {t for t, c in header_counter.items() if c / total_pages >= freq_threshold}
    footer_texts = {t for t, c in footer_counter.items() if c / total_pages >= freq_threshold}
    return header_texts, footer_texts

def filter_header_footer(lines, H, header_texts, footer_texts, top_ratio=0.10, bottom_ratio=0.10):
    top_y = H * top_ratio
    bot_y = H * (1 - bottom_ratio)
    out = []
    for ln in lines:
        t = norm_text(ln["text"])
        if not t:
            continue
        if is_page_number(t) and (ln["y1"] <= top_y or ln["y0"] >= bot_y):
            continue
        if t in header_texts and ln["y1"] <= top_y:
            continue
        if t in footer_texts and ln["y0"] >= bot_y:
            continue
        out.append(ln)
    return out


# ----------------------------
# Step 3: Column detection (x0 clustering via histogram peaks)
# ----------------------------

def detect_columns(lines, page_width, min_lines=25):
    """
    Returns column x-ranges: list of (col_left, col_right)
    Heuristic:
      - if not enough lines => single column
      - build histogram of x0
      - find 1 or 2 dominant peaks separated by a gutter
    """
    if len(lines) < min_lines:
        return [(0.0, page_width)]

    x0s = [ln["x0"] for ln in lines]
    # Bin histogram
    bins = 60
    bw = page_width / bins
    hist = [0] * bins
    for x in x0s:
        bi = int(max(0, min(bins - 1, x // bw)))
        hist[bi] += 1

    # Find top peaks
    peaks = sorted(range(bins), key=lambda i: hist[i], reverse=True)[:6]
    peaks = sorted(peaks)

    # Helper: merge nearby peaks
    merged = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > 3:
            merged.append(p)
    # If only one strong region => 1 column
    if len(merged) <= 1:
        return [(0.0, page_width)]

    # Try to pick two best-separated peaks
    # Choose two peaks with highest counts and separation
    candidates = []
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            sep = abs(merged[j] - merged[i])
            score = hist[merged[i]] + hist[merged[j]] + sep
            candidates.append((score, merged[i], merged[j]))
    candidates.sort(reverse=True)
    _, p1, p2 = candidates[0]
    if p1 > p2:
        p1, p2 = p2, p1

    # Estimate split (gutter) between p1 and p2 by finding low-density valley
    lo, hi = p1, p2
    valley = min(range(lo, hi + 1), key=lambda k: hist[k])
    split_x = (valley + 0.5) * bw

    # Validate gutter exists: valley relatively low compared to peaks
    if hist[valley] > 0.6 * max(hist[p1], hist[p2]):
        return [(0.0, page_width)]

    # Build columns: left [0, split], right [split, W]
    # Add margins to avoid mis-assign
    margin = 0.01 * page_width
    return [(0.0, split_x + margin), (split_x - margin, page_width)]


# ----------------------------
# Step 4: Reading order + block typing
# ----------------------------

def assign_column(ln, columns):
    x_mid = (ln["x0"] + ln["x1"]) / 2
    for idx, (cl, cr) in enumerate(columns):
        if cl <= x_mid <= cr:
            return idx
    # fallback: nearest
    dists = [min(abs(x_mid - cl), abs(x_mid - cr)) for (cl, cr) in columns]
    return int(min(range(len(columns)), key=lambda i: dists[i]))

def is_spanning_title(ln, page_width, min_width_ratio=0.75):
    return (ln["x1"] - ln["x0"]) >= page_width * min_width_ratio

def sort_reading_order(lines, columns, page_width):
    """
    Return lines ordered in reading order:
      - spanning titles first by y
      - then each column by y, columns left->right
    """
    spanning = []
    regular = []
    for ln in lines:
        if is_spanning_title(ln, page_width) and ln["y0"] < 0.35 * 99999:  # no-op, keep heuristic simple
            spanning.append(ln)
        else:
            regular.append(ln)

    spanning.sort(key=lambda ln: (ln["y0"], ln["x0"]))

    # group regular by column
    cols = defaultdict(list)
    for ln in regular:
        c = assign_column(ln, columns)
        cols[c].append(ln)
    for c in cols:
        cols[c].sort(key=lambda ln: (ln["y0"], ln["x0"]))

    ordered = []
    # Insert spanning titles in correct y-position relative to columns:
    # simplest: merge by y
    merged_regular = []
    for c in range(len(columns)):
        merged_regular.extend(cols.get(c, []))
    merged_regular.sort(key=lambda ln: (ln["y0"], assign_column(ln, columns), ln["x0"]))

    # true merge: spanning + merged_regular by y
    i = j = 0
    while i < len(spanning) and j < len(merged_regular):
        if spanning[i]["y0"] <= merged_regular[j]["y0"]:
            ordered.append(spanning[i]); i += 1
        else:
            ordered.append(merged_regular[j]); j += 1
    ordered.extend(spanning[i:])
    ordered.extend(merged_regular[j:])
    return ordered


# ----------------------------
# Step 5: Reflow to paragraphs + heading detection + list detection
# ----------------------------

def compute_size_bands(all_lines, k=4):
    """
    Map font sizes to heading levels.
    Take top distinct sizes as headings, rest as body.
    """
    sizes = [ln["size"] for ln in all_lines if ln["size"] > 0]
    if not sizes:
        return {}
    # cluster by rounding
    rounded = sorted({round(s, 1) for s in sizes}, reverse=True)
    # take top k as potential headings
    top = rounded[:k]
    bands = {}
    for idx, s in enumerate(top):
        # idx 0 biggest => H1
        bands[s] = idx + 1
    return bands

def heading_level(ln, size_bands):
    s = round(ln["size"], 1)
    if s in size_bands:
        # Make sure it's not too long; headings are usually short
        if len(ln["text"]) <= 140:
            return size_bands[s]
    return None

def merge_lines_to_blocks(ordered_lines, page_width):
    """
    Merge into blocks: heading, list item, paragraph.
    """
    blocks = []
    prev = None

    def flush_prev():
        nonlocal prev
        if prev:
            blocks.append(prev)
            prev = None

    for ln in ordered_lines:
        text = ln["text"]
        # Ignore obvious artifacts
        if is_page_number(text):
            continue

        # Heading?
        # We'll assign later with size bands; keep for now
        # List?
        if starts_like_list_item(text):
            flush_prev()
            blocks.append({"type": "list_item", "text": strip_list_marker(text), "meta": ln})
            continue

        # Paragraph merge heuristic
        if prev and prev["type"] == "para":
            # Determine if this line continues the paragraph
            dy = ln["y0"] - prev["meta"]["y1"]
            same_col = abs(ln["x0"] - prev["meta"]["x0"]) <= 12  # px-ish
            # If line spacing small and alignment similar => merge
            if dy <= max(6.0, 0.9 * ln["size"]) and (same_col or ln["x0"] > prev["meta"]["x0"]):
                # hyphenation
                prev["text"] = hyphenation_join(prev["text"], text)
                # extend meta bbox
                prev["meta"]["x0"] = min(prev["meta"]["x0"], ln["x0"])
                prev["meta"]["y0"] = min(prev["meta"]["y0"], ln["y0"])
                prev["meta"]["x1"] = max(prev["meta"]["x1"], ln["x1"])
                prev["meta"]["y1"] = max(prev["meta"]["y1"], ln["y1"])
                continue
            else:
                flush_prev()

        # start new paragraph
        if not prev:
            prev = {"type": "para", "text": text, "meta": dict(ln)}  # copy meta
        else:
            # shouldn't happen due to flush_prev
            prev["text"] = hyphenation_join(prev["text"], text)
            prev["meta"]["y1"] = max(prev["meta"]["y1"], ln["y1"])

    flush_prev()
    return blocks

def blocks_to_markdown(blocks, size_bands):
    md_lines = []
    for b in blocks:
        if b["type"] == "list_item":
            md_lines.append(f"- {b['text']}")
            continue

        # heading detection from meta
        lvl = heading_level(b["meta"], size_bands)
        if lvl is not None:
            hashes = "#" * min(6, lvl)
            md_lines.append(f"{hashes} {b['text']}")
        else:
            md_lines.append(b["text"])

        md_lines.append("")  # blank line between blocks
    # remove excessive blank lines
    out = []
    blank = 0
    for line in md_lines:
        if line.strip() == "":
            blank += 1
            if blank <= 2:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip() + "\n"


# ----------------------------
# Main conversion
# ----------------------------

def pdf_to_markdown(pdf_path: str, max_pages=None):
    doc = fitz.open(pdf_path)

    all_pages_lines = []
    page_heights = []
    page_widths = []

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        page_heights.append(page.rect.height)
        page_widths.append(page.rect.width)
        lines = extract_lines(page)
        all_pages_lines.append(lines)

    header_texts, footer_texts = detect_repeated_header_footer(
        all_pages_lines, page_heights, top_ratio=0.10, bottom_ratio=0.10, freq_threshold=0.6
    )

    all_blocks = []
    all_lines_flat = []

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        W = page_widths[i]
        H = page_heights[i]
        lines = filter_header_footer(all_pages_lines[i], H, header_texts, footer_texts, top_ratio=0.10, bottom_ratio=0.10)
        columns = detect_columns(lines, W)

        ordered = sort_reading_order(lines, columns, W)
        all_lines_flat.extend(ordered)

        blocks = merge_lines_to_blocks(ordered, W)
        # mark page boundaries lightly (optional)
        # all_blocks.append({"type": "para", "text": f"<!-- Page {i+1} -->", "meta": {"size": 0, "text": ""}})
        all_blocks.extend(blocks)

    size_bands = compute_size_bands(all_lines_flat, k=4)
    return blocks_to_markdown(all_blocks, size_bands)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf2md.py input.pdf [max_pages]", file=sys.stderr)
        sys.exit(1)
    pdf_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    md = pdf_to_markdown(pdf_path, max_pages=max_pages)
    print(md, end="")
