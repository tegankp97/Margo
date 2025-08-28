import os
import zipfile
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_DIR = 'Slop'

# --------------------------- Patterns & Config ---------------------------
PASSIVE_VOICE_RE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE)
ADVERB_RE = re.compile(r"\b\w+ly\b", re.IGNORECASE)
TRIGRAM_RE = re.compile(r"(\b\w+\b)\s+(\b\w+\b)\s+(\b\w+\b)")
# Negation pivot: Not X…Not Y (1–3 words each), flexible punctuation
PIVOT_RE = re.compile(
    r'(?i)\bnot\s+(?:\w+(?:\s+\w+){0,2})\b[\s\-\–—,;:]*\bnot\s+(?:\w+(?:\s+\w+){0,2})\b'
)
# Exact phrases
BAW_RE = re.compile(r"(?i)\bbarely above a whisper\b")
INSIGHT_RE = re.compile(r"(?i)\bprovide(?:s)? a valuable insight\b")

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
WORD_SPLIT = re.compile(r"\w+")
URL_RE = re.compile(r'https?://\S+|www\.\S+|\S+@\S+')

# Basic English stopwords (small, enough for n-gram filtering)
STOPWORDS = set('''a an the and or but if while though although for nor so yet of in on at by to from as with about into over after before between without within across under above below up down out off over again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very can will just don should now he she they it we you i him her them his hers their its our your me us my mine ours yours himself herself themselves itself ourselves yourselves'''.split())

# Files/sections to exclude (front/back matter, nav, ads, etc.)
EXCLUDE_FILE_HINTS = (
    'toc', 'nav', 'title', 'copyright', 'dedication', 'acknowledge',
    'about the author', 'cover', 'back matter', 'advertise', 'subscribe', 'newsletter',
    # extra NF boilerplate to drop early
    'reference', 'bibliograph', 'index', 'glossary', 'appendix', 'notes', 'footnote',
    'works cited', 'figures', 'tables'
)

# Metadata generator strings that imply AI
GENERATOR_KEYWORDS = (
    'gpt', 'openai', 'claude', 'novelai', 'sudowrite', 'writesonic', 'rytr'
)

# Thresholds
DEFAULT_THRESHOLD = 14
PREFIX_THRESHOLD = 10  # for csp_, dra_, drs_
LENGTH_FLOOR = 25000  # words for structural/recurrence checks
LEX_DIV_MIN_WORDS = 35000
DEBUG = False  # temporary: show per-criterion scores for each title

# --------------------------- Utilities ---------------------------

def normalize_quotes_spaces(s: str) -> str:
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = URL_RE.sub(' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def is_all_caps(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)

def has_digit_token(tri):
    return any(any(ch.isdigit() for ch in w) for w in tri)

# --------------------------- Extraction ---------------------------

def extract_segments_and_text(z):
    """Return (segments_texts, full_text) after crude filtering of junk files/lines."""
    segments = []
    names = [n for n in z.namelist() if n.lower().endswith(('.xhtml', '.html', '.htm'))]
    for name in names:
        lname = name.lower()
        if any(h in lname for h in EXCLUDE_FILE_HINTS):
            continue
        try:
            raw = z.read(name)
            text = BeautifulSoup(raw, 'html.parser').get_text('\n')
            # Line-level filtering: drop ultra-short lines, all-caps lines, and pure chapter headers
            filtered_lines = []
            for ln in text.splitlines():
                ln = normalize_quotes_spaces(ln)
                if not ln:
                    continue
                if is_all_caps(ln) and len(ln) > 3:
                    continue
                if len(ln.split()) < 4:
                    continue
                if re.match(r'(?i)^chapter\b', ln):
                    continue
                filtered_lines.append(ln)
            clean = '\n'.join(filtered_lines)
            if clean.strip():
                segments.append(clean)
        except Exception:
            continue
    full = '\n'.join(segments)
    return segments, full

# --------------------------- Metadata & NF detection ---------------------------

def check_meta_ai(z):
    """Return (flag, reasons, subjects) if metadata suggests AI authorship (ChatGPT/Generator)."""
    reasons = []
    subjects = []
    opf_path = ''
    try:
        data = z.read('META-INF/container.xml')
        root = ET.fromstring(data)
        rf = root.find('.//{*}rootfile')
        if rf is not None:
            opf_path = rf.attrib.get('full-path', '')
    except Exception:
        pass

    candidates = []
    if opf_path:
        candidates.append(opf_path)
    candidates += [n for n in z.namelist() if n.lower().endswith('.opf') and n != opf_path]

    ns = {'dc': 'http://purl.org/dc/elements/1.1/', 'opf': 'http://www.idpf.org/2007/opf'}
    for p in candidates:
        try:
            opf_data = z.read(p)
            r = ET.fromstring(opf_data)
            fields = []
            # creators/publishers
            for tag in ('creator', 'publisher'):
                for el in r.findall(f'.//dc:{tag}', ns):
                    if el is not None and el.text:
                        fields.append((tag, el.text.strip()))
            # subjects
            for el in r.findall('.//dc:subject', ns):
                if el is not None and el.text:
                    subjects.append(el.text.strip())
            # generic meta
            for el in r.findall('.//{*}meta'):
                name = (el.attrib.get('name', '') or '').lower()
                content = (el.attrib.get('content', '') or '').strip()
                if name in ('author', 'creator', 'publisher', 'generator', 'subject') and content:
                    fields.append((name, content))
                    if name == 'subject':
                        subjects.append(content)
            # Evaluate
            for key, val in fields:
                v = val.lower()
                if 'chatgpt' in v or 'chat gpt' in v:
                    reasons.append(f"metadata {key}='{val}'")
                if key == 'generator' and any(k in v for k in GENERATOR_KEYWORDS):
                    reasons.append(f"metadata generator='{val}'")
            if reasons:
                return True, reasons, subjects
        except Exception:
            continue
    return False, [], subjects

NF_SUBJECT_HINTS = {
    'business','self-help','self help','history','science','technology','tech',
    'education','reference','study guide','textbook','politics','economics',
    'psychology','management','marketing','finance','biography','memoir'
}

NF_SECTION_HEADINGS = (
    'introduction','abstract','methodology','methods','references',
    'bibliography','notes','appendix','glossary','index','works cited'
)

CITATION_PATTERNS = [
    re.compile(r'\((?:19|20)\d{2}\)'),       # (1999), (2015)
    re.compile(r'\bet al\.' , re.IGNORECASE),
    re.compile(r'\bibid\.' , re.IGNORECASE),
    re.compile(r'\[\d+\]'),
    re.compile(r'\bdoi:\s*\S+', re.IGNORECASE),
]

def detect_nonfiction(subjects, full_text, total_words):
    """Return (nf_mode: bool, nf_reasons: list[str], nf_signals_count: int)."""
    nf_reasons = []
    signals = 0

    # Subject hints
    subj_str = ' | '.join(s.lower() for s in subjects)
    if any(h in subj_str for h in NF_SUBJECT_HINTS):
        signals += 1
        nf_reasons.append('nf_subject')

    # Section headings present (any two distinct)
    found = set()
    for h in NF_SECTION_HEADINGS:
        if re.search(rf'(?mi)^\s*{re.escape(h)}\b', full_text):
            found.add(h)
    if len(found) >= 2:
        signals += 1
        nf_reasons.append('nf_sections')

    # Citation markers density per 5k words
    cit_hits = 0
    for pat in CITATION_PATTERNS:
        cit_hits += len(pat.findall(full_text))
    per5k = cit_hits / max(1, (total_words / 5000))
    if per5k >= 8:
        signals += 1
        nf_reasons.append('nf_citations')

    # List / figure / table structure density
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    bullet_like = 0
    figtab = 0
    for ln in lines:
        if re.match(r'^\s*(?:[-*•]|[0-9]{1,2}\.)\s+', ln):
            bullet_like += 1
        if re.search(r'\b(?:figure|table)\s+\d+\b', ln, flags=re.IGNORECASE):
            figtab += 1
    structural = (bullet_like + figtab) / max(1, len(lines))
    if structural > 0.04:
        signals += 1
        nf_reasons.append('nf_lists_figures')

    nf_mode = signals >= 2
    return nf_mode, nf_reasons, signals

def detect_copyright_pre_2022(full_text):
    """Return (dampen: bool, year: int|None)."""
    hits = []
    for m in re.finditer(r'(?:©|\(c\)|copyright)\s*(?:\([cC]\))?\s*(?:\D{0,5})\b(19|20)\d{2}\b', full_text, flags=re.IGNORECASE):
        y = int(m.group(0)[-4:])
        hits.append(y)
    if hits:
        yr = min(hits)
        if yr <= 2021:
            return True, yr
    return False, None

# --------------------------- Text metrics ---------------------------

def analyze_tropes(text, segments):
    words = WORD_SPLIT.findall(text.lower())
    total_words = len(words)
    if total_words == 0:
        return {}

    # Lexical diversity
    unique = len(set(words))
    lex_div = unique / total_words

    # Sentence metrics
    sentences_raw = [s for s in SENT_SPLIT.split(text) if s.strip()]
    sentences_norm = [normalize_quotes_spaces(s) for s in sentences_raw]  # keep case for NF filters
    sent_lengths = [len(WORD_SPLIT.findall(s)) for s in sentences_raw]
    avg_sent = sum(sent_lengths) / len(sent_lengths) if sentences_raw else 0
    var_sent = (sum((l - avg_sent)**2 for l in sent_lengths) / len(sent_lengths)) if sentences_raw else 0

    # Passive and adverb usage
    passive_ratio = len(PASSIVE_VOICE_RE.findall(text)) / total_words
    adverb_ratio = len(ADVERB_RE.findall(text)) / total_words

    # Trigram repetition density (content trigrams only)
    raw_trigrams = TRIGRAM_RE.findall(text.lower())
    content_trigrams = [t for t in raw_trigrams if sum(1 for w in t if w in STOPWORDS) < 2]
    tri_counts = Counter(content_trigrams)
    repeat_tris = sum(1 for cnt in tri_counts.values() if cnt > 1)
    tri_density = (repeat_tris / len(content_trigrams)) if content_trigrams else 0

    # NF-specific: digit-free content trigrams density
    content_trigrams_nf = [t for t in content_trigrams if not has_digit_token(t)]
    tri_counts_nf = Counter(content_trigrams_nf)
    repeat_tris_nf = sum(1 for cnt in tri_counts_nf.values() if cnt > 1)
    tri_density_nf = (repeat_tris_nf / len(content_trigrams_nf)) if content_trigrams_nf else 0

    # Phrase/pivot counts
    baw_count = len(BAW_RE.findall(text))
    insight_count = len(INSIGHT_RE.findall(text))
    pivot_count = len(PIVOT_RE.findall(text))

    # 1) Exact-duplicate sentence detector (ignore short & headers & NF boilerplate)
    dup_candidates = []
    for s, s_len in zip(sentences_norm, sent_lengths):
        s_norm = s.strip()
        if s_len < 7:
            continue
        if s_norm.lower().startswith('chapter '):
            continue
        # NF noise filters
        if re.search(r'\b(figure|table)\s+\d+\b', s_norm, flags=re.IGNORECASE):
            continue
        if 'doi:' in s_norm.lower() or 'http' in s_norm.lower():
            continue
        if sum(ch.isdigit() for ch in s_norm) >= 2:
            continue
        if re.match(r'^\s*(section\s+\d+(\.\d+)*)', s_norm, flags=re.IGNORECASE):
            continue
        if re.match(r'^\s*\d+(\.\d+)*\s+\S+', s_norm):  # numbered outline headings
            continue
        dup_candidates.append(normalize_quotes_spaces(s_norm.lower()))
    sent_counter = Counter(dup_candidates)
    dup_total = sum(cnt for cnt in sent_counter.values() if cnt > 1)
    dup_ratio = (dup_total / len(dup_candidates)) if dup_candidates else 0

    # 2) Segment-length uniformity (by words per segment), exclude tiny segments
    seg_word_lengths = []
    for s in segments:
        wl = len(WORD_SPLIT.findall(s))
        if wl >= 250:
            seg_word_lengths.append(wl)
    seg_cv = 1.0
    if len(seg_word_lengths) >= 15:
        mean_len = sum(seg_word_lengths) / len(seg_word_lengths)
        if mean_len > 0:
            var = sum((l - mean_len) ** 2 for l in seg_word_lengths) / len(seg_word_lengths)
            std = var ** 0.5
            seg_cv = std / mean_len

    # 3) Windowed lexical diversity (flatness): 2k window, 1k step
    window, step = 2000, 1000
    ttrs = []
    if total_words >= window:
        for i in range(0, total_words - window + 1, step):
            w_slice = words[i:i + window]
            ttrs.append(len(set(w_slice)) / window)
    ttr_var = 0.0
    if ttrs:
        mean_ttr = sum(ttrs) / len(ttrs)
        ttr_var = sum((x - mean_ttr) ** 2 for x in ttrs) / len(ttrs)

    # 5) Distant 6-gram recurrence (freq >=3 and spaced >2000 words, contenty grams)
    recurring_far_6grams = 0
    recurring_far_6grams_nf = 0
    if total_words >= LENGTH_FLOOR:
        positions = defaultdict(list)
        positions_nf = defaultdict(list)
        for i in range(total_words - 5):
            gram = tuple(words[i:i+6])
            non_stop = sum(1 for w in gram if w not in STOPWORDS)
            avg_tok_len = sum(len(w) for w in gram) / 6
            if non_stop >= 3 and avg_tok_len >= 4:
                positions[gram].append(i)
            # NF stricter: no digits & ≥4 non-stopwords
            if non_stop >= 4 and avg_tok_len >= 4 and all(not any(ch.isdigit() for ch in w) for w in gram):
                positions_nf[gram].append(i)
        for pos_list in positions.values():
            if len(pos_list) >= 3:
                avg_spacing = (pos_list[-1] - pos_list[0]) / (len(pos_list) - 1)
                if avg_spacing > 2000:
                    recurring_far_6grams += 1
        for pos_list in positions_nf.values():
            if len(pos_list) >= 3:
                avg_spacing = (pos_list[-1] - pos_list[0]) / (len(pos_list) - 1)
                if avg_spacing > 2000:
                    recurring_far_6grams_nf += 1

    return {
        'total_words': total_words,
        'lex_div': lex_div,
        'avg_sent': avg_sent,
        'var_sent': var_sent,
        'passive_ratio': passive_ratio,
        'adverb_ratio': adverb_ratio,
        'tri_density': tri_density,
        'tri_density_nf': tri_density_nf,
        'baw_count': baw_count,
        'insight_count': insight_count,
        'pivot_count': pivot_count,
        'dup_total': dup_total,
        'dup_ratio': dup_ratio,
        'seg_cv': seg_cv,
        'ttr_var': ttr_var,
        'ttr_windows': len(ttrs),
        'recurring_far_6grams': recurring_far_6grams,
        'recurring_far_6grams_nf': recurring_far_6grams_nf,
    }

# --------------------------- Scoring ---------------------------

def score_ai(m, nf_mode=False, copyright_dampen=None, nf_reasons=None):
    score = 0
    reasons = []
    breakdown = []  # (criterion, points)

    if nf_mode:
        reasons.append("mode=non-fiction (damped structural penalties)")
        if nf_reasons:
            reasons.extend(nf_reasons)

    # "barely above a whisper" (cap total +6)
    baw = m.get('baw_count', 0)
    baw_pts = min(6, 3 * baw) if baw > 0 else 0
    if baw_pts:
        reasons.append(f"'barely above a whisper'×{baw} (+{baw_pts})")
    breakdown.append(("baw_phrase", baw_pts))

    # "provide(s) a valuable insight" (cap total +6; cap +3 in NF)
    ins = m.get('insight_count', 0)
    if nf_mode:
        ins_pts = min(3, 3 * ins) if ins > 0 else 0
    else:
        ins_pts = min(6, 3 * ins) if ins > 0 else 0
    if ins_pts:
        reasons.append(f"\"provide(s) a valuable insight\"×{ins} (+{ins_pts})")
    breakdown.append(("insight_phrase", ins_pts))

    # Negation pivots
    pv = m.get('pivot_count', 0)
    pv_trigger = 8 if nf_mode else 4
    pv_pts = 2 if pv > pv_trigger else 0
    if pv_pts:
        reasons.append(f"neg_pivot {pv}>{pv_trigger} (+{pv_pts})")
    breakdown.append(("negation_pivots", pv_pts))

    # Apply structural/recurrence checks only for longer books
    long_enough = m.get('total_words', 0) >= LENGTH_FLOOR

    # 1) Duplicate sentences – tiered by ratio & count (stricter in NF)
    dup_pts = 0
    if long_enough:
        dt = m.get('dup_total', 0)
        dr = m.get('dup_ratio', 0)
        if nf_mode:
            # stricter tiers for NF
            if dt >= 200 and dr > 0.22: dup_pts = 10
            elif dt >= 150 and dr > 0.18: dup_pts = 8
            elif dt >= 100 and dr > 0.15: dup_pts = 6
            elif dt >= 75  and dr > 0.12: dup_pts = 5
            elif dt >= 50  and dr > 0.10: dup_pts = 3
            elif dt >= 35  and dr > 0.08: dup_pts = 1
        else:
            if dt >= 150 and dr > 0.20: dup_pts = 10
            elif dt >= 100 and dr > 0.15: dup_pts = 8
            elif dt >= 75  and dr > 0.12: dup_pts = 6
            elif dt >= 50  and dr > 0.10: dup_pts = 5
            elif dt >= 35  and dr > 0.08: dup_pts = 4
            elif dt >= 25  and dr > 0.06: dup_pts = 2
    if dup_pts:
        reasons.append(f"duplicate_sentences {m['dup_total']} ({m['dup_ratio']:.2%}) (+{dup_pts})")
    breakdown.append(("duplicate_sentences", dup_pts))

    # 2) Segment-length uniformity – tiered by CV (stricter in NF)
    seg_pts = 0
    if long_enough:
        cv = m.get('seg_cv', 1.0)
        if nf_mode:
            if cv < 0.10: seg_pts = 6
            elif cv < 0.13: seg_pts = 5
            elif cv < 0.16: seg_pts = 4
            elif cv < 0.18: seg_pts = 3
            elif cv < 0.20: seg_pts = 2
        else:
            if cv < 0.12: seg_pts = 6
            elif cv < 0.15: seg_pts = 5
            elif cv < 0.18: seg_pts = 4
            elif cv < 0.20: seg_pts = 3
            elif cv < 0.22: seg_pts = 2
    if seg_pts:
        reasons.append(f"segment_cv {m['seg_cv']:.2f} (+{seg_pts})")
    breakdown.append(("segment_uniformity", seg_pts))

    # 3) TTR flatness – tiered by variance (gated by dup or seg)
    ttr_pts = 0
    if long_enough and m.get('ttr_windows', 0) >= 12 and ((dup_pts > 0) or (seg_pts > 0)):
        tv = m.get('ttr_var', 1.0)
        if tv < 1e-4: ttr_pts = 6
        elif tv < 2e-4: ttr_pts = 5
        elif tv < 3e-4: ttr_pts = 4
        elif tv < 4e-4: ttr_pts = 3
        elif tv < 5e-4: ttr_pts = 2
    if ttr_pts:
        reasons.append(f"ttr_var {m['ttr_var']:.6f} (+{ttr_pts})")
    breakdown.append(("ttr_flatness", ttr_pts))

    # 4) Distant 6-grams – tiered scoring (stricter & digit-free in NF)
    grams_pts = 0
    if long_enough:
        g = m.get('recurring_far_6grams_nf', 0) if nf_mode else m.get('recurring_far_6grams', 0)
        if nf_mode:
            if g >= 115: grams_pts = 12
            elif g >= 90: grams_pts = 10
            elif g >= 75: grams_pts = 8
            elif g >= 60: grams_pts = 6
            elif g >= 45: grams_pts = 3
            elif g >= 35: grams_pts = 1
        else:
            if g >= 100: grams_pts = 12
            elif g >= 75: grams_pts = 10
            elif g >= 60: grams_pts = 8
            elif g >= 45: grams_pts = 6
            elif g >= 30: grams_pts = 3
            elif g >= 25: grams_pts = 1
    if grams_pts:
        reasons.append(f"recurring_far_6grams {g} (+{grams_pts})")
    breakdown.append(("recurring_6grams", grams_pts))

    # 5) Lexical diversity (keep your single-tier @ <0.13 with 35k+ words)
    lex_cond = m.get('total_words', 0) >= LEX_DIV_MIN_WORDS and m['lex_div'] < 0.13
    lex_pts = 6 if lex_cond else 0
    if lex_pts:
        reasons.append(f"lex_div {m['lex_div']:.2f}<0.13 (+{lex_pts})")
    breakdown.append(("lexical_diversity", lex_pts))

    # 6) Sentence variance (you set very low threshold already)
    var_pts = 4 if m['var_sent'] < 15 else 0
    if var_pts:
        reasons.append(f"var_sent {m['var_sent']:.1f}<15 (+{var_pts})")
    breakdown.append(("sentence_variance", var_pts))

    # 7) Passive/adverbs (loosen in NF)
    passive_thr = 0.08 if nf_mode else 0.05
    adverb_thr  = 0.12 if nf_mode else 0.09

    passive_pts = 1 if m['passive_ratio'] > passive_thr else 0
    if passive_pts:
        reasons.append(f"passive_ratio {m['passive_ratio']:.2f}>{passive_thr:.2f} (+{passive_pts})")
    breakdown.append(("passive_ratio", passive_pts))

    adverb_pts = 1 if m['adverb_ratio'] > adverb_thr else 0
    if adverb_pts:
        reasons.append(f"adverb_ratio {m['adverb_ratio']:.2f}>{adverb_thr:.2f} (+{adverb_pts})")
    breakdown.append(("adverb_ratio", adverb_pts))

    # 8) Trigram density – tiered; stricter & digit-free in NF
    td = m.get('tri_density_nf' if nf_mode else 'tri_density', 0)
    if nf_mode:
        if td > 0.11: trigram_pts = 10
        elif td > 0.09: trigram_pts = 8
        elif td > 0.07: trigram_pts = 5
        elif td > 0.05: trigram_pts = 4
        elif td > 0.04: trigram_pts = 3
        else: trigram_pts = 0
    else:
        if td > 0.10: trigram_pts = 10
        elif td > 0.08: trigram_pts = 8
        elif td > 0.06: trigram_pts = 5
        elif td > 0.04: trigram_pts = 4
        elif td > 0.03: trigram_pts = 3
        else: trigram_pts = 0
    if trigram_pts:
        reasons.append(f"tri_density {td:.2f} (+{trigram_pts})")
    breakdown.append(("trigram_density", trigram_pts))

    # Sum points
    score = sum(pts for _, pts in breakdown)

    # Copyright gentle scoring: reduce by 2 points if year <= 2021
    if copyright_dampen:
        score = max(0, score - 2)
        breakdown.append(("copyright_gentle", -2))
        reasons.append(f"copyright<=2021 (gentle -2)")

    return score, reasons, breakdown

# --------------------------- Pipeline ---------------------------

def process_epub(path):
    fn = os.path.basename(path)
    try:
        with zipfile.ZipFile(path) as z:
            # metadata check (auto QUALITY) + subjects for NF
            meta_flag, meta_reasons, subjects = check_meta_ai(z)
            if meta_flag:
                return fn, 'QUALITY WARNING', 'score=auto(metadata)', meta_reasons, []
            segments, full = extract_segments_and_text(z)
    except Exception:
        return fn, 'ERROR', 'No Text', [], []

    metrics = analyze_tropes(full, segments)
    if not metrics:
        return fn, 'NO TEXT', 'No Text', [], []

    # Detect NF mode & copyright damping
    nf_mode, nf_markers, _ = detect_nonfiction(subjects, full, metrics['total_words'])
    copy_dampen, copy_year = detect_copyright_pre_2022(full)
    if copy_dampen:
        nf_markers.append(f'copyright_year={copy_year}')

    score, reasons, breakdown = score_ai(metrics, nf_mode=nf_mode, copyright_dampen=copy_dampen, nf_reasons=nf_markers)

    # thresholds
    lower = fn.lower()
    threshold = PREFIX_THRESHOLD if lower.startswith(('csp_', 'dra_', 'drs_')) else DEFAULT_THRESHOLD
    label = 'QUALITY WARNING' if score >= threshold else 'HUMAN-LIKELY'
    summary = f"score={score} threshold={threshold}"
    return fn, label, summary, reasons, breakdown

def main():
    results = []
    epub_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith('.epub')]
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(process_epub, p) for p in epub_files]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Print HUMAN-LIKELY first
    for fn, label, summary, reasons, breakdown in results:
        if label == 'HUMAN-LIKELY':
            print(f"{fn}: {label} ({summary})")
    # Then QUALITY WARNING
    for fn, label, summary, reasons, breakdown in results:
        if label == 'QUALITY WARNING':
            extra = f" – {', '.join(reasons)}" if reasons else ''
            print(f"{fn}: {label} ({summary}){extra}")
    # Finally others
    for fn, label, summary, reasons, breakdown in results:
        if label not in ('HUMAN-LIKELY','QUALITY WARNING'):
            print(f"{fn}: {label} ({summary})")

    # --- DEBUG per-criterion scores (set DEBUG=False to hide) ---
    if DEBUG:
        print("--- DEBUG PER-CRITERION SCORES ---")
        for fn, label, summary, reasons, breakdown in results:
            print(f"{fn}: {label} ({summary})")
            for key, pts in breakdown:
                print(f"  {key}: {pts:+d}")

if __name__ == '__main__':
    main()
