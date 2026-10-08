import re
import io
import html
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
# PAGE CONFIG
st.set_page_config( page_title="Accessify", page_icon="🛠️",layout="wide",initial_sidebar_state="expanded")
# STYLING
st.markdown("""
<style>
.mk-watermark {
    position: fixed;
    bottom: 8px;
    right: 12px;
    font-size: 10px;
    opacity: 0.12;
    z-index: 9999;
    pointer-events: none;
}
</style>

<div class="mk-watermark">MK</div>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .hero { padding: 2rem 2.2rem; border-radius: 26px; background: linear-gradient(135deg,#111827 0%,#312e81 48%,#0f766e 100%); color:#fff; box-shadow:0 18px 45px rgba(15,23,42,.25); margin-bottom:1.4rem; }
    .brand { font-size:3.8rem; font-weight:950; letter-spacing:.08em; line-height:1; }
    .project-name { font-size:1.45rem; font-weight:800; margin-top:.5rem; }
    .hero-desc { font-size:1.05rem; max-width:1000px; line-height:1.6; margin-top:1rem; color:#f8fafc; }
    .score-wrap { border-radius:28px; padding:1.5rem; color:#fff; text-align:center; background:linear-gradient(145deg,#030712,#111827); border:3px solid #facc15; box-shadow:0 15px 35px rgba(0,0,0,.3); }
    .score-number { font-size:5.2rem; font-weight:950; line-height:1; letter-spacing:-.05em; }
    .score-label { font-size:1.15rem; font-weight:900; text-transform:uppercase; letter-spacing:.08em; }
    .score-bar { height:16px; border-radius:99px; background:#374151; overflow:hidden; margin-top:1rem; }
    .score-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#ef4444,#facc15,#22c55e); }
    .issue-card { padding:1rem 1.2rem; border-radius:14px; border:1px solid rgba(128,128,128,.25); margin-bottom:.8rem; }
    .small-muted { opacity:.68; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
# CONSTANTS
DATASET_NAME = "Original_full_data.csv"
DEFAULT_DATASET_PATHS = [
    DATASET_NAME,
    "./data/Original_full_data.csv",
    "/mnt/data/Original_full_data.csv",
]
SEVERITY_WEIGHTS = {
    5: 5.0,  # critical
    4: 4.0,  # serious
    3: 3.0,  # moderate
    2: 2.0,  # minor
    1: 1.0,
}
# DATASET LOADING
@st.cache_data(show_spinner=False)
def load_accessguru_dataset(path):
    df = pd.read_csv(path, low_memory=False)
    # Normalize column names
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]
    required = [
        "violationnumberID",
        "initialImpactScore",
        "Description",
        "AffectedHTMLElement(s)",
        "category",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The AccessGuru CSV is missing expected columns: "
            + ", ".join(missing)
        )
    df["initialImpactScore"] = pd.to_numeric(
        df["initialImpactScore"],
        errors="coerce",
    )
    df = df.dropna(subset=["initialImpactScore"]).copy()
    # Build text used by the ML model
    text_columns = [
        "violationnumberID",
        "Description",
        "AffectedHTMLElement(s)",
        "category",
        "LongDescription",
        "updated_Description",
    ]
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["model_text"] = (
        df["violationnumberID"]
        + " "
        + df["Description"]
        + " "
        + df["AffectedHTMLElement(s)"]
        + " "
        + df["category"]
        + " "
        + df["LongDescription"]
    )
    return df
def find_dataset():
    for path in DEFAULT_DATASET_PATHS:
        try:
            with open(path, "rb"):
                return path
        except Exception:
            pass
    return None
# ML MODEL
@st.cache_resource(show_spinner=False)
def train_accessguru_model(df):
    """
    Learn the relationship between an accessibility violation's
    textual/HTML description and its AccessGuru impact score.

    This is deliberately interpretable and lightweight:
        TF-IDF -> Ridge regression

    The live scanner then uses semantic similarity against the
    AccessGuru corpus to retrieve the closest known violations and
    their fixes.
    """
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(df["model_text"])
    y = df["initialImpactScore"].astype(float).values
    model = Ridge(alpha=2.0)
    model.fit(X, y)
    return vectorizer, model
@st.cache_resource(show_spinner=False)
def build_dataset_search_index(df):
    """
    Separate TF-IDF index used to find the closest AccessGuru
    examples for a detected problem.
    """

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(df["model_text"])

    return vectorizer, X
# WEB FETCHING
def normalize_url(url):
    url = url.strip()
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.netloc:
        return None
    return url
def fetch_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Accessify/1.0; "
            "+https://example.com/accessibility)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if "text/html" not in content_type and response.text.count("<") < 10:
        raise ValueError(
            "The URL did not return a normal HTML webpage."
        )

    return response.text, response.url
# ACCESSIBILITY HELPERS
def visible_text(element):
    return element.get_text(" ", strip=True)
def has_accessible_name(element, soup):
    """
    Practical accessible-name heuristic.

    It checks common sources:
      - visible text
      - aria-label
      - aria-labelledby
      - title
      - alt
      - associated label
    """
    if element.get("aria-label", "").strip():
        return True
    labelledby = element.get("aria-labelledby", "").strip()
    if labelledby:
        ids = labelledby.split()
        if any(soup.find(id=i) for i in ids):
            return True
    if element.get("title", "").strip():
        return True
    if element.get("alt", "").strip():
        return True
    if visible_text(element):
        return True
    element_id = element.get("id")
    if element_id:
        label = soup.find(
            "label",
            attrs={"for": element_id},
        )
        if label and visible_text(label):
            return True
    parent_label = element.find_parent("label")
    if parent_label and visible_text(parent_label):
        return True
    return False
def _contrast_ratio(foreground, background):
    """Return WCAG contrast ratio for two RGB colors."""
    def lum(rgb):
        vals = []
        for c in rgb:
            c = c / 255.0
            vals.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]
    a, b = lum(foreground), lum(background)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)
def _parse_css_color(value):
    """Parse common CSS hex/rgb/named colors used in inline styles."""
    if not value:
        return None
    v = str(value).strip().lower()
    named = {
        'white': (255,255,255), 'black': (0,0,0), 'red': (255,0,0),
        'blue': (0,0,255), 'green': (0,128,0), 'gray': (128,128,128),
        'grey': (128,128,128), 'yellow': (255,255,0), 'orange': (255,165,0),
        'purple': (128,0,128), 'navy': (0,0,128), 'transparent': None,
    }
    if v in named:
        return named[v]
    m = re.fullmatch(r'#([0-9a-f]{3}|[0-9a-f]{6})', v)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = ''.join(c*2 for c in h)
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))
    m = re.fullmatch(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*[\d.]+)?\s*\)', v)
    if m:
        return tuple(int(m.group(i)) for i in (1,2,3))
    return None


def _extract_style_color(style, property_name):
    if not style:
        return None
    m = re.search(rf'{re.escape(property_name)}\s*:\s*([^;]+)', style, re.I)
    return _parse_css_color(m.group(1)) if m else None


def _specific_issue_details(issue):
    """Turn a generic rule finding into presentation-ready, concrete evidence."""
    raw = html.unescape(issue.get('element', ''))
    soup = BeautifulSoup(raw, 'lxml')
    el = soup.find()
    tag = el.name if el else 'element'
    attrs = dict(el.attrs) if el else {}
    text = visible_text(el) if el else ''
    text = re.sub(r'\s+', ' ', text).strip()
    rule = issue.get('rule', '')

    details = {
        'detected_value': '',
        'why_it_matters': 'This can create a barrier for people using screen readers, keyboard navigation, magnification or other assistive technology.',
        'specific_fix': issue.get('description', ''),
    }
    if rule == 'image-alt':
        details['detected_value'] = 'alt=' + repr(attrs.get('alt'))
        details['why_it_matters'] = 'A screen reader may announce the image as an unlabeled graphic, so a blind user may miss information conveyed visually.'
        details['specific_fix'] = 'If informative, replace the missing/empty alt with a concise description of the image. If purely decorative, keep alt="" intentionally.'
    elif rule == 'button-name':
        details['detected_value'] = f'<button> text={text!r}, aria-label={attrs.get("aria-label")!r}'
        details['why_it_matters'] = 'A screen-reader user can reach the control but may hear only “button” without knowing what action it performs.'
        details['specific_fix'] = 'Give the button a visible text label, or an aria-label that states the action (for example, “Close menu”).'
    elif rule == 'link-name':
        details['detected_value'] = f'<a href={attrs.get("href")!r}> accessible text={text!r}'
        details['why_it_matters'] = 'A screen-reader user navigating by links cannot determine where this link goes.'
        details['specific_fix'] = 'Use concise, meaningful link text such as “Download annual report” instead of an empty or icon-only link.'
    elif rule == 'label':
        details['detected_value'] = f'{tag} id={attrs.get("id")!r}, name={attrs.get("name")!r}, placeholder={attrs.get("placeholder")!r}'
        details['why_it_matters'] = 'Without a programmatic label, a screen reader may announce a form field without telling the user what information to enter.'
        details['specific_fix'] = 'Add a <label for="..."> connected to the control id, or provide a correct accessible name with aria-label/aria-labelledby.'
    elif rule == 'html-has-lang':
        details['detected_value'] = f'<html lang={attrs.get("lang")!r}'
        details['why_it_matters'] = 'Screen readers use the page language to choose pronunciation rules and voice settings.'
        details['specific_fix'] = 'Set the correct language, for example <html lang="en"> or <html lang="ta">.'
    elif rule == 'document-title':
        details['detected_value'] = '<title> is missing or empty'
        details['why_it_matters'] = 'Users of screen readers and browser tabs need a clear page title to identify the current page.'
        details['specific_fix'] = 'Add a unique, descriptive <title>, such as <title>Admissions — ABC University</title>.'
    elif rule == 'heading-order':
        details['detected_value'] = f'Found heading <{tag}>{text}</{tag}>'
        details['why_it_matters'] = 'Screen-reader users often navigate by headings; skipped levels can make the document structure confusing.'
        details['specific_fix'] = issue.get('description', '') + ' Change the heading level to reflect the actual content hierarchy.'
    elif rule == 'frame-title':
        details['detected_value'] = f'iframe title={attrs.get("title")!r}'
        details['why_it_matters'] = 'An iframe is a separate browsing context; without a title, a screen-reader user cannot tell what it contains.'
        details['specific_fix'] = 'Add a short title describing the embedded content, e.g. title="Campus map".'
    elif rule == 'video-caption':
        details['detected_value'] = 'No <track kind="captions"> detected inside this <video>.'
        details['why_it_matters'] = 'People who are deaf or hard of hearing can miss spoken information without synchronized captions.'
        details['specific_fix'] = 'Add a WebVTT captions track with kind="captions" and the correct srclang.'
    elif rule == 'duplicate-id':
        details['detected_value'] = attrs.get('id', '')
        details['why_it_matters'] = 'Duplicate IDs can break label associations, ARIA references and scripted focus behavior.'
        details['specific_fix'] = f'Give each element a unique id. For example, change repeated id="{attrs.get("id", "value")}" to distinct IDs.'
    elif rule == 'aria-hidden-focus':
        details['detected_value'] = 'aria-hidden="true" contains a focusable descendant'
        details['why_it_matters'] = 'Keyboard users can focus something that screen-reader users are told does not exist, creating an inconsistent experience.'
        details['specific_fix'] = 'Remove aria-hidden from the interactive region, or ensure hidden content is genuinely non-focusable.'
    elif rule == 'tabindex':
        details['detected_value'] = f'tabindex={attrs.get("tabindex")!r}'
        details['why_it_matters'] = 'Positive tabindex values can force an unexpected keyboard order that differs from the visual/document order.'
        details['specific_fix'] = 'Remove the positive tabindex and rely on natural DOM order; use tabindex="0" only when needed.'
    elif rule == 'empty-table-header':
        details['detected_value'] = f'<th> text={text!r}'
        details['why_it_matters'] = 'A blank table header makes it harder for screen-reader users to understand row/column relationships.'
        details['specific_fix'] = 'Give the header a meaningful label such as “Price”, “Date”, or “Status”.'
    elif rule == 'no-autoplay-audio':
        details['detected_value'] = '<' + tag + ' autoplay>'
        details['why_it_matters'] = 'Unexpected audio can interfere with screen-reader speech and can be disorienting for users with cognitive or sensory disabilities.'
        details['specific_fix'] = 'Remove autoplay or ensure users have an immediate, keyboard-accessible pause/mute control.'
    elif rule == 'contrast':
        details['detected_value'] = f'Foreground {issue.get("foreground")} on background {issue.get("background")} → contrast {issue.get("contrast_ratio")}:1'
        details['why_it_matters'] = 'Low contrast can make text difficult or impossible to read for people with low vision or color-vision differences.'
        details['specific_fix'] = 'Increase the contrast between foreground and background. WCAG AA generally requires 4.5:1 for normal text and 3:1 for large text.'
    issue.update(details)
    return issue


def add_contrast_issues(soup, issues):
    """Detect low-contrast pairs when explicit inline colors are available."""
    seen = set()
    for el in soup.find_all(style=True):
        style = el.get('style', '')
        fg = _extract_style_color(style, 'color')
        bg = _extract_style_color(style, 'background-color') or _extract_style_color(style, 'background')
        if not fg or not bg:
            continue
        ratio = _contrast_ratio(fg, bg)
        text = re.sub(r'\s+', ' ', visible_text(el)).strip()
        if not text or ratio >= 4.5:
            continue
        key = (str(el)[:500], round(ratio, 2))
        if key in seen:
            continue
        seen.add(key)
        severity = 5 if ratio < 3 else 4
        issue = {
            'rule': 'contrast', 'severity': severity,
            'title': 'Text has insufficient color contrast',
            'description': f'This element explicitly sets {fg} text on {bg} background, producing a {ratio:.2f}:1 contrast ratio.',
            'element': str(el)[:1500], 'wcag': 'WCAG 1.4.3 Contrast (Minimum)',
            'category': 'Visual Contrast', 'foreground': fg, 'background': bg,
            'contrast_ratio': round(ratio, 2),
        }
        issues.append(_specific_issue_details(issue))


def add_issue(
    issues,
    rule,
    severity,
    title,
    description,
    element,
    wcag="",
    category="Syntax",
):
    issue = {
        "rule": rule,
        "severity": int(severity),
        "title": title,
        "description": description,
        "element": str(element)[:1500],
        "wcag": wcag,
        "category": category,
    }
    issues.append(_specific_issue_details(issue))


# ============================================================
# RULE-BASED ACCESSIBILITY SCANNER
# ============================================================

def scan_page(html_text):
    soup = BeautifulSoup(html_text, "lxml")

    issues = []

    # --------------------------------------------------------
    # 1. Images
    # --------------------------------------------------------

    for img in soup.find_all("img"):

        alt = img.get("alt")

        if alt is None:
            add_issue(
                issues,
                "image-alt",
                5,
                "Image is missing alternative text",
                (
                    "A screen reader may not know what this image "
                    "represents. Add meaningful alt text, or use "
                    'alt="" when the image is purely decorative.'
                ),
                img,
                "WCAG 1.1.1 Non-text Content",
            )

        elif alt.strip() == "" and not (
            img.get("role") == "presentation"
            or img.get("aria-hidden") == "true"
        ):
            # Empty alt can be valid for decorative images, so make
            # this a lower-severity warning.
            add_issue(
                issues,
                "image-alt",
                3,
                "Image has empty alternative text",
                (
                    "Check whether this image is decorative. If it "
                    "conveys information, provide a meaningful alt."
                ),
                img,
                "WCAG 1.1.1 Non-text Content",
            )

    # --------------------------------------------------------
    # 2. Input type=image
    # --------------------------------------------------------

    for inp in soup.find_all(
        "input",
        attrs={"type": re.compile("^image$", re.I)},
    ):
        if not has_accessible_name(inp, soup):
            add_issue(
                issues,
                "input-image-alt",
                5,
                "Image input has no accessible name",
                "Add an alt attribute or another accessible name.",
                inp,
                "WCAG 1.1.1 Non-text Content",
            )

    # --------------------------------------------------------
    # 3. Buttons
    # --------------------------------------------------------

    for button in soup.find_all("button"):

        if not has_accessible_name(button, soup):

            add_issue(
                issues,
                "button-name",
                5,
                "Button has no accessible name",
                (
                    "Screen-reader users need to know what the "
                    "button does. Add visible text or aria-label."
                ),
                button,
                "WCAG 4.1.2 Name, Role, Value",
            )

    # --------------------------------------------------------
    # 4. Links
    # --------------------------------------------------------

    for link in soup.find_all("a"):

        href = link.get("href")

        if href is not None and not has_accessible_name(link, soup):

            add_issue(
                issues,
                "link-name",
                5,
                "Link has no discernible text",
                (
                    "Add meaningful link text or an accessible "
                    "aria-label describing the destination."
                ),
                link,
                "WCAG 2.4.4 Link Purpose",
            )

    # --------------------------------------------------------
    # 5. Form controls
    # --------------------------------------------------------

    form_controls = soup.find_all(
        ["input", "textarea", "select"]
    )

    for control in form_controls:

        input_type = (
            control.get("type", "").lower()
        )

        if input_type in [
            "hidden",
            "submit",
            "button",
            "reset",
        ]:
            continue

        if not has_accessible_name(control, soup):

            add_issue(
                issues,
                "label",
                5,
                "Form control has no accessible label",
                (
                    "Associate this form control with a label "
                    "or provide an appropriate accessible name."
                ),
                control,
                "WCAG 1.3.1 Info and Relationships",
            )

    # --------------------------------------------------------
    # 6. HTML language
    # --------------------------------------------------------

    html_tag = soup.find("html")

    if html_tag:

        lang = html_tag.get("lang", "").strip()

        if not lang:

            add_issue(
                issues,
                "html-has-lang",
                4,
                "Page does not specify a language",
                (
                    'Add a language attribute such as '
                    '<html lang="en">.'
                ),
                html_tag,
                "WCAG 3.1.1 Language of Page",
            )

    # --------------------------------------------------------
    # 7. Page title
    # --------------------------------------------------------

    title = soup.find("title")

    if not title or not visible_text(title):

        add_issue(
            issues,
            "document-title",
            4,
            "Page is missing a meaningful title",
            (
                "Add a descriptive <title> so screen-reader and "
                "browser users can identify the page."
            ),
            title if title else soup,
            "WCAG 2.4.2 Page Titled",
        )

    # --------------------------------------------------------
    # 8. Main landmark
    # --------------------------------------------------------

    if not soup.find("main"):

        # ARIA main is also accepted.
        main_role = soup.find(
            attrs={"role": "main"}
        )

        if not main_role:

            add_issue(
                issues,
                "landmark",
                3,
                "No main content landmark detected",
                (
                    "Consider wrapping the primary content in "
                    "<main> so assistive technologies can navigate "
                    "the page structure more efficiently."
                ),
                soup.body if soup.body else soup,
                "WCAG 1.3.1 Info and Relationships",
            )

    # --------------------------------------------------------
    # 9. Navigation landmark
    # --------------------------------------------------------

    if not soup.find("nav") and not soup.find(
        attrs={"role": "navigation"}
    ):

        add_issue(
            issues,
            "navigation",
            2,
            "No navigation landmark detected",
            (
                "Consider using <nav> for major navigation regions."
            ),
            soup.body if soup.body else soup,
            "WCAG 1.3.1 Info and Relationships",
        )

    # --------------------------------------------------------
    # 10. Iframes
    # --------------------------------------------------------

    for frame in soup.find_all("iframe"):

        if not has_accessible_name(frame, soup):

            add_issue(
                issues,
                "frame-title",
                4,
                "Iframe has no accessible title",
                (
                    "Add a descriptive title attribute explaining "
                    "the iframe's content."
                ),
                frame,
                "WCAG 4.1.2 Name, Role, Value",
            )

    # --------------------------------------------------------
    # 11. Video captions
    # --------------------------------------------------------

    for video in soup.find_all("video"):

        tracks = video.find_all("track")

        has_captions = any(
            str(t.get("kind", "")).lower() == "captions"
            for t in tracks
        )

        if not has_captions:

            add_issue(
                issues,
                "video-caption",
                5,
                "Video has no detected captions track",
                (
                    "Provide captions for spoken dialogue and "
                    "important audio information."
                ),
                video,
                "WCAG 1.2.2 Captions",
            )

    # --------------------------------------------------------
    # 12. Duplicate IDs
    # --------------------------------------------------------

    ids = {}

    for element in soup.find_all(attrs={"id": True}):

        element_id = element.get("id")

        if not element_id:
            continue

        ids.setdefault(element_id, []).append(element)

    for duplicate_id, elements in ids.items():

        if len(elements) > 1:

            add_issue(
                issues,
                "duplicate-id",
                4,
                "Duplicate HTML id detected",
                (
                    f'The id "{duplicate_id}" appears '
                    f"{len(elements)} times. IDs should be unique."
                ),
                elements[0],
                "WCAG 4.1.1 Parsing",
            )

    # --------------------------------------------------------
    # 13. aria-hidden + focusable content
    # --------------------------------------------------------

    for hidden in soup.find_all(
        attrs={"aria-hidden": "true"}
    ):

        focusable = hidden.find(
            [
                "a",
                "button",
                "input",
                "select",
                "textarea",
                "iframe",
            ]
        )

        tabindex = hidden.get("tabindex")

        if focusable is not None or (
            tabindex is not None
            and tabindex not in ["-1", ""]
        ):

            add_issue(
                issues,
                "aria-hidden-focus",
                5,
                "Hidden content contains focusable content",
                (
                    'Do not place focusable controls inside an '
                    'aria-hidden="true" region.'
                ),
                hidden,
                "WCAG 4.1.2 Name, Role, Value",
            )

    # --------------------------------------------------------
    # 14. Positive tabindex
    # --------------------------------------------------------

    for element in soup.find_all(
        attrs={"tabindex": True}
    ):

        try:
            tabindex = int(element.get("tabindex"))
        except Exception:
            continue

        if tabindex > 0:

            add_issue(
                issues,
                "tabindex",
                3,
                "Positive tabindex detected",
                (
                    "Avoid positive tabindex values. Prefer the "
                    "natural document order and tabindex='0' only "
                    "when an element genuinely needs focus."
                ),
                element,
                "WCAG 2.4.3 Focus Order",
            )

    # --------------------------------------------------------
    # 15. Heading hierarchy
    # --------------------------------------------------------

    headings = soup.find_all(
        re.compile("^h[1-6]$")
    )

    previous_level = None

    for heading in headings:

        level = int(heading.name[1])

        if (
            previous_level is not None
            and level > previous_level + 1
        ):

            add_issue(
                issues,
                "heading-order",
                3,
                "Heading hierarchy skips a level",
                (
                    f"The page jumps from H{previous_level} "
                    f"to H{level}. Use a logical heading hierarchy."
                ),
                heading,
                "WCAG 1.3.1 Info and Relationships",
            )

        previous_level = level

    # --------------------------------------------------------
    # 16. Empty table headers
    # --------------------------------------------------------

    for th in soup.find_all("th"):

        if not has_accessible_name(th, soup):

            add_issue(
                issues,
                "empty-table-header",
                3,
                "Table header has no discernible text",
                (
                    "Give table headers meaningful text so screen "
                    "reader users can understand the table structure."
                ),
                th,
                "WCAG 1.3.1 Info and Relationships",
            )

    # --------------------------------------------------------
    # 17. Autoplay audio/video
    # --------------------------------------------------------

    for media in soup.find_all(
        ["audio", "video"]
    ):

        if media.has_attr("autoplay"):

            add_issue(
                issues,
                "no-autoplay-audio",
                3,
                "Media is configured to autoplay",
                (
                    "Avoid unexpected audio. Give users an obvious "
                    "mechanism to pause, stop or mute it."
                ),
                media,
                "WCAG 1.4.2 Audio Control",
            )

    # --------------------------------------------------------
    # 18. ARIA hidden body
    # --------------------------------------------------------

    if soup.body and soup.body.get("aria-hidden") == "true":

        add_issue(
            issues,
            "aria-hidden-body",
            5,
            "Document body is aria-hidden",
            (
                "The main document body should not normally be "
                'removed from the accessibility tree.'
            ),
            soup.body,
            "WCAG 4.1.2 Name, Role, Value",
        )

    # --------------------------------------------------------
    # 19. Color contrast (explicit inline CSS heuristic)
    # --------------------------------------------------------
    add_contrast_issues(soup, issues)

    # --------------------------------------------------------
    # 20. Role=text heuristic
    # --------------------------------------------------------

    for element in soup.find_all(
        attrs={"role": "text"}
    ):

        if element.find(
            [
                "button",
                "a",
                "input",
                "select",
                "textarea",
            ]
        ):

            add_issue(
                issues,
                "aria-text",
                4,
                "role=text is used around interactive content",
                (
                    "Review this ARIA usage and avoid masking "
                    "interactive descendants from assistive "
                    "technologies."
                ),
                element,
                "WCAG 4.1.2 Name, Role, Value",
            )

    return soup, issues


# ============================================================
# ML ENRICHMENT
# ============================================================

def enrich_issues_with_dataset(
    issues,
    df,
    model_vectorizer,
    impact_model,
    search_vectorizer,
    search_matrix,
):
    if not issues:
        return issues

    query_texts = []

    for issue in issues:
        query_texts.append(
            " ".join(
                [
                    issue["rule"],
                    issue["title"],
                    issue["description"],
                    issue["element"],
                    issue["wcag"],
                    issue["category"],
                ]
            )
        )

    # ML-predicted impact
    X_query = model_vectorizer.transform(query_texts)
    predicted = impact_model.predict(X_query)

    # Dataset semantic similarity
    Q = search_vectorizer.transform(query_texts)
    similarities = cosine_similarity(
        Q,
        search_matrix,
    )

    for i, issue in enumerate(issues):

        # Clamp to AccessGuru's normal 1-5 scale
        ml_score = float(
            np.clip(predicted[i], 1.0, 5.0)
        )

        best_indices = np.argsort(
            similarities[i]
        )[-3:][::-1]

        best_idx = int(best_indices[0])

        dataset_row = df.iloc[best_idx]

        issue["ml_impact"] = round(
            ml_score,
            2,
        )

        issue["dataset_similarity"] = round(
            float(similarities[i][best_idx]),
            3,
        )

        issue["dataset_rule"] = str(
            dataset_row.get(
                "violationnumberID",
                "",
            )
        )

        issue["dataset_description"] = str(
            dataset_row.get(
                "Description",
                "",
            )
        )

        issue["dataset_category"] = str(
            dataset_row.get(
                "category",
                "",
            )
        )

        issue["dataset_fix"] = str(
            dataset_row.get(
                "fixed_code_desc",
                "",
            )
        )

        issue["dataset_wcag"] = str(
            dataset_row.get(
                "updated_Description",
                "",
            )
        )

        issue["dataset_source_url"] = str(
            dataset_row.get(
                "webURL",
                "",
            )
        )

    return issues


# ============================================================
# SCORE
# ============================================================

def calculate_accessibility_score(issues, html_text):
    """
    Explainable 0-100 WCAG-oriented screening score.

    The score is calculated from the automated rule checks themselves.
    AccessGuru/ML impact is NOT used to create the percentage.

    Each applicable rule has a fixed maximum weight. A finding loses part of
    that rule's weight according to how many applicable elements fail and the
    scanner's fixed severity. The function also returns a breakdown so the UI
    can show exactly where points were lost.
    """
    soup = BeautifulSoup(html_text, "lxml")

    weights = {
        "image-alt": 15.0,
        "input-image-alt": 3.0,
        "button-name": 10.0,
        "link-name": 10.0,
        "label": 15.0,
        "html-has-lang": 5.0,
        "document-title": 5.0,
        "landmark": 4.0,
        "navigation": 2.0,
        "frame-title": 4.0,
        "video-caption": 7.0,
        "duplicate-id": 5.0,
        "aria-hidden-focus": 5.0,
        "tabindex": 3.0,
        "heading-order": 4.0,
        "empty-table-header": 3.0,
        "no-autoplay-audio": 3.0,
        "aria-hidden-body": 2.0,
        "contrast": 10.0,
        "aria-text": 3.0,
    }

    page_level_rules = {
        "html-has-lang", "document-title", "landmark", "navigation",
        "duplicate-id", "aria-hidden-body"
    }

    opportunities = {
        "image-alt": len(soup.find_all("img")),
        "input-image-alt": len(soup.find_all("input", attrs={"type": re.compile("^image$", re.I)})),
        "button-name": len(soup.find_all("button")),
        "link-name": sum(1 for x in soup.find_all("a") if x.get("href") is not None),
        "label": sum(
            1 for x in soup.find_all(["input", "textarea", "select"])
            if x.get("type", "").lower() not in ["hidden", "submit", "button", "reset"]
        ),
        "html-has-lang": 1 if soup.find("html") else 0,
        "document-title": 1,
        "landmark": 1,
        "navigation": 1,
        "frame-title": len(soup.find_all("iframe")),
        "video-caption": len(soup.find_all("video")),
        "duplicate-id": 1 if soup.find_all(attrs={"id": True}) else 0,
        "aria-hidden-focus": len(soup.find_all(attrs={"aria-hidden": "true"})),
        "tabindex": len(soup.find_all(attrs={"tabindex": True})),
        "heading-order": max(0, len(soup.find_all(re.compile("^h[1-6]$"))) - 1),
        "empty-table-header": len(soup.find_all("th")),
        "no-autoplay-audio": len(soup.find_all(["audio", "video"])),
        "aria-hidden-body": 1 if soup.body else 0,
        "contrast": sum(
            1 for x in soup.find_all(style=True)
            if _extract_style_color(x.get("style", ""), "color")
            and (_extract_style_color(x.get("style", ""), "background-color")
                 or _extract_style_color(x.get("style", ""), "background"))
            and visible_text(x)
        ),
        "aria-text": len(soup.find_all(attrs={"role": "text"})),
    }

    applicable_weight = 0.0
    total_loss = 0.0
    breakdown = []

    for rule, weight in weights.items():
        opportunity_count = opportunities.get(rule, 0)
        if opportunity_count <= 0:
            continue

        applicable_weight += weight
        rule_issues = [x for x in issues if x.get("rule") == rule]
        failed_count = len(rule_issues)

        if not rule_issues:
            continue

        if rule in page_level_rules:
            failed_ratio = 1.0
        else:
            failed_ratio = min(1.0, failed_count / float(opportunity_count))

        severity_factor = sum(
            min(5.0, max(1.0, float(x.get("severity", 1)))) / 5.0
            for x in rule_issues
        ) / max(1, failed_count)

        raw_rule_loss = weight * failed_ratio * severity_factor
        score_loss = (100.0 * raw_rule_loss / max(weight, 0.0001))
        # score_loss is the percentage of the rule's available contribution.
        # Normalize it later against all applicable rule weights.
        normalized_loss = 100.0 * raw_rule_loss
        total_loss += normalized_loss

        title = str(rule_issues[0].get("title", rule.replace("-", " ").title()))
        wcag = str(rule_issues[0].get("wcag", ""))
        severity_value = max(float(x.get("severity", 1)) for x in rule_issues)
        severity_text = severity_name(severity_value)

        breakdown.append({
            "rule": rule,
            "title": title,
            "wcag": wcag,
            "affected": failed_count,
            "opportunities": opportunity_count,
            "severity": severity_text,
            "weight": weight,
            "raw_loss": raw_rule_loss,
            "severity_factor": severity_factor,
        })

    if applicable_weight <= 0:
        return 100.0, []

    # Convert weighted rule losses into a 0-100 score.
    score = 100.0 - (100.0 * sum(x["raw_loss"] for x in breakdown) / applicable_weight)
    score = round(max(0.0, min(100.0, score)), 1)

    # Store the actual percentage-point contribution of every finding group.
    for item in breakdown:
        item["points_lost"] = round(
            100.0 * item["raw_loss"] / applicable_weight, 1
        )

    breakdown.sort(key=lambda x: x["points_lost"], reverse=True)
    return score, breakdown


def score_label(score):
    if score >= 85:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Needs Improvement"
    if score >= 35:
        return "Poor"
    return "Critical"


def severity_name(score):
    score = float(score)

    if score >= 4.5:
        return "Critical"
    if score >= 3.5:
        return "Serious"
    if score >= 2.5:
        return "Moderate"
    return "Minor"


# ============================================================
# REPORT
# ============================================================

def make_report(url, score, issues, features):
    lines = []

    lines.append("ACCESSIFY — AUTOMATED ACCESSIBILITY REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"URL: {url}")
    lines.append(f"Accessibility Screening Score: {score}/100")
    lines.append(f"Rating: {score_label(score)}")
    lines.append("")
    lines.append("NOTE:")
    lines.append(
        "This is an automated screening estimate, not a WCAG "
        "conformance certification."
    )
    lines.append("")
    lines.append("PAGE SUMMARY")
    lines.append("-" * 60)

    for key, value in features.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("DETECTED ISSUES")
    lines.append("-" * 60)

    if not issues:
        lines.append("No automated issues detected.")

    for i, issue in enumerate(
        sorted(
            issues,
            key=lambda x: -float(
                x.get(
                    "ml_impact",
                    x["severity"],
                )
            ),
        ),
        1,
    ):

        impact = issue.get(
            "ml_impact",
            issue["severity"],
        )

        lines.append(
            f"{i}. {issue['title']} "
            f"[{severity_name(impact)}]"
        )

        lines.append(
            f"   Rule: {issue['rule']}"
        )

        lines.append(
            f"   WCAG: {issue['wcag']}"
        )

        lines.append(f"   What is wrong: {issue.get('description', '')}")
        lines.append(f"   Detected value: {issue.get('detected_value', '')}")
        lines.append(f"   Why it matters: {issue.get('why_it_matters', '')}")
        lines.append(f"   Specific fix: {issue.get('specific_fix', '')}")
        lines.append(f"   Element: {issue['element']}")

        if issue.get("dataset_fix"):
            lines.append(
                "   Dataset-grounded recommendation:"
            )
            lines.append(
                "   "
                + issue["dataset_fix"][:2000]
                .replace("\n", " ")
            )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# UI HEADER
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="brand">🛠️ ACCESSIFY</div>'
    '<div class="project-name">Automated accessibility engine</div>'
    '<div class="hero-desc">An Automated Auditing tool that instantly scans digital environments to detect critical accessibility flaws, such as missing structural tags and poor visual contrast and helps the visually impaired to access the internet better.</div>'
    '</div>',
    unsafe_allow_html=True,
)


# Official WCAG reference
st.markdown(
    '<div style="margin:-0.5rem 0 1rem 0;">'
    '<a href="https://www.w3.org/WAI/standards-guidelines/wcag/" target="_blank" style="text-decoration:none;">'
    '<button style="padding:0.6rem 1rem;border-radius:10px;border:1px solid #6366f1;background:#eef2ff;color:#312e81;font-weight:800;cursor:pointer;">📖 Learn about WCAG rules ↗</button>'
    '</a></div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛠️ Accessify")

    st.write(
        """
        Accessify helps developers discover accessibility
        barriers that can affect people using screen readers,
        keyboard navigation and other assistive technologies.
        """
    )

    st.divider()

    st.subheader("Technology")

    st.write(
        """
        • Python + BeautifulSoup  
        • Scikit-learn ML  
        • TF-IDF semantic matching  
        • AccessGuru accessibility dataset  
        • WCAG-oriented rule checks  
        • Streamlit dashboard
        """
    )

    st.divider()

    st.subheader("Important")

    st.caption(
        "Automated accessibility testing cannot prove that a "
        "website is fully accessible. Manual testing and testing "
        "with people who use assistive technologies are still important."
    )


# ============================================================
# DATASET INITIALIZATION
# ============================================================

dataset_path = find_dataset()

if not dataset_path:

    st.error(
        f"Could not find `{DATASET_NAME}`."
    )

    st.info(
        "Place Original_full_data.csv in the same folder as app.py "
        "and restart Streamlit."
    )

    st.stop()

try:

    with st.spinner(
        "Loading AccessGuru accessibility dataset..."
    ):

        dataset = load_accessguru_dataset(
            dataset_path
        )

        model_vectorizer, impact_model = (
            train_accessguru_model(dataset)
        )

        search_vectorizer, search_matrix = (
            build_dataset_search_index(dataset)
        )

except Exception as e:

    st.error(
        "The dataset could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DATASET INFO
# ============================================================

with st.expander(
    "📊 Dataset Used",
    expanded=False,
):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(dataset):,}",
    )

    c2.metric(
        "Violation Types",
        dataset["violationnumberID"].nunique(),
    )

    c3.metric(
        "Categories",
        dataset["category"].nunique(),
    )

    c4.metric(
        "Domains",
        dataset["domain"].nunique()
        if "domain" in dataset.columns
        else "—",
    )

    st.caption(
        "The uploaded AccessGuru dataset is actively used by the ML layer: "
        "Accessify trains a TF-IDF + Ridge impact model and performs semantic retrieval against real-world accessibility violation records."
    )


# ============================================================
# URL INPUT
# ============================================================

st.markdown("## 🔎 Analyze a website")

url_input = st.text_input(
    "Website URL",
    placeholder="https://example.com",
    label_visibility="collapsed",
)

analyze = st.button(
    "🚀 Analyze Website",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    url = normalize_url(url_input)

    if not url:

        st.warning(
            "Please enter a valid website URL."
        )

        st.stop()

    try:

        with st.spinner(
            "Fetching webpage..."
        ):

            page_html, final_url = fetch_page(url)

        with st.spinner(
            "Scanning accessibility barriers..."
        ):

            soup, issues = scan_page(
                page_html
            )

        with st.spinner(
            "Applying AccessGuru ML model..."
        ):

            issues = enrich_issues_with_dataset(
                issues,
                dataset,
                model_vectorizer,
                impact_model,
                search_vectorizer,
                search_matrix,
            )

        score, score_breakdown = calculate_accessibility_score(
            issues,
            page_html,
        )

        # ----------------------------------------------------
        # Summary features
        # ----------------------------------------------------

        images = soup.find_all("img")

        buttons = soup.find_all("button")

        inputs = soup.find_all(
            ["input", "textarea", "select"]
        )

        links = soup.find_all("a")

        summary = {
            "Images": len(images),
            "Buttons": len(buttons),
            "Form controls": len(inputs),
            "Links": len(links),
            "Detected issues": len(issues),
        }

        # Store in session state for download / rerender.
        st.session_state["analysis"] = {
            "url": final_url,
            "score": score,
            "score_breakdown": score_breakdown,
            "issues": issues,
            "summary": summary,
            "report": make_report(
                final_url,
                score,
                issues,
                summary,
            ),
        }

    except requests.exceptions.Timeout:

        st.error(
            "The website took too long to respond."
        )

        st.stop()

    except requests.exceptions.RequestException as e:

        st.error(
            "Could not access the website."
        )

        st.code(str(e))

        st.stop()

    except Exception as e:

        st.error(
            "Something went wrong during analysis."
        )

        st.exception(e)

        st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    score = result["score"]
    score_breakdown = result.get("score_breakdown", [])
    issues = result["issues"]
    summary = result["summary"]

    st.divider()

    st.markdown("## 📈 Accessibility Results")

    score_col, stats_col = st.columns(
        [1, 2]
    )

    with score_col:

        st.markdown(
            f"""
            <div class="score-wrap">
                <div style="font-weight:800;opacity:.8">ACCESSIFY ACCESSIBILITY SCORE</div>
                <div class="score-number">{score}</div>
                <div class="score-label">{score_label(score)}</div>
                <div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>
                <div style="margin-top:.7rem;font-size:.82rem;opacity:.78">100 = strongest automated result</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("WCAG-oriented automated screening score — not an official WCAG conformance percentage or certification.")

        # ----------------------------------------------------
        # EXPLAINABLE SCORE BREAKDOWN
        # ----------------------------------------------------
        st.markdown("### 🧮 Why did points get lost?")
        st.caption("Accessify starts at 100.0. The rows below show exactly which automated checks caused the score to fall.")

        if not score_breakdown:
            st.success("No applicable automated accessibility failures were found, so no points were deducted.")
        else:
            displayed_loss = 0.0
            for item in score_breakdown:
                loss = float(item["points_lost"])
                displayed_loss += loss
                wcag_text = f" · {item['wcag']}" if item["wcag"] else ""
                affected_text = (
                    f"{item['affected']} affected"
                    if item['affected'] == 1
                    else f"{item['affected']} affected"
                )
                st.markdown(
                    f"**−{loss:.1f} points · {item['title']}**  "
                    f"  \n{affected_text} of {item['opportunities']} opportunities · {item['severity']}{wcag_text}"
                )

            st.markdown(
                f"**Score calculation: 100.0 − {displayed_loss:.1f} = {score:.1f}**"
            )

            with st.expander("📘 How Accessify calculates the score"):
                st.markdown(
                    """
                    **1. Start at 100 points.**

                    **2. Only applicable checks count.** A rule such as video captions is not counted when the page contains no video.

                    **3. Each WCAG-oriented rule has a fixed weight.** More important automated checks contribute more to the score than minor checks.

                    **4. Repeated failures are counted by affected elements.** One missing image alternative is not treated the same as dozens of missing alternatives.

                    **5. Scanner severity adjusts the loss.** The fixed rule severity is used for the score; the AccessGuru/ML impact prediction is kept separate for prioritization.

                    **6. The final score is normalized to 0–100.** Every deduction shown above is a percentage-point contribution to the displayed score.

                    This is an **automated WCAG-oriented screening score**, not a claim that a website is officially WCAG conformant.
                    """
                )

    with stats_col:

        cols = st.columns(5)

        for i, (key, value) in enumerate(
            summary.items()
        ):

            cols[i].metric(
                key,
                value,
            )

    # --------------------------------------------------------
    # Overall message
    # --------------------------------------------------------

    if score >= 90:

        st.success(
            "🎉 The automated scanner found relatively few "
            "accessibility barriers."
        )

    elif score >= 75:

        st.info(
            "👍 The page has a reasonable accessibility baseline, "
            "but some barriers should still be fixed."
        )

    elif score >= 60:

        st.warning(
            "⚠️ Several accessibility barriers were detected. "
            "Prioritize the serious and critical findings."
        )

    else:

        st.error(
            "🚨 Significant accessibility barriers were detected. "
            "The page should undergo accessibility remediation."
        )

    # --------------------------------------------------------
    # Issue distribution
    # --------------------------------------------------------

    st.markdown("## 🚨 Detected Problems")

    if not issues:

        st.success(
            "No automated accessibility issues were detected."
        )

    else:

        issue_df = pd.DataFrame(
            [
                {
                    "What is wrong": x.get("detected_value") or x["title"],
                    "Evidence / element": re.sub(
                        r"\s+",
                        " ",
                        html.unescape(x.get("element", "")),
                    )[:220],
                    "Why it matters": x.get("why_it_matters", ""),
                    "Specific fix": x.get("specific_fix", x.get("description", "")),
                    "Severity": severity_name(
                        x.get("ml_impact", x["severity"])
                    ),
                    "ML impact": round(
                        x.get("ml_impact", x["severity"]),
                        2,
                    ),
                }
                for x in issues
            ]
        )

        issue_df = issue_df.sort_values(
            "ML impact",
            ascending=False,
        )

        st.dataframe(
            issue_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Detailed recommendations
        # ----------------------------------------------------

        st.markdown(
            "## 🛠 Developer Recommendations"
        )

        sorted_issues = sorted(
            issues,
            key=lambda x: -float(
                x.get(
                    "ml_impact",
                    x["severity"],
                )
            ),
        )

        for number, issue in enumerate(
            sorted_issues,
            1,
        ):

            impact = issue.get(
                "ml_impact",
                issue["severity"],
            )

            severity = severity_name(
                impact
            )

            with st.expander(
                f"{number}. {issue['title']} "
                f"— {severity}"
            ):

                left, right = st.columns(
                    [1, 1]
                )

                with left:

                    st.markdown(
                        f"**Detected rule:** `{issue['rule']}`"
                    )

                    st.markdown(
                        f"**WCAG area:** "
                        f"{issue['wcag'] or 'General accessibility'}"
                    )

                    st.markdown(
                        f"**Category:** {issue['category']}"
                    )
                    st.markdown(
                        f"**ML-predicted impact:** "
                        f"`{impact:.2f}/5`"
                    )
                    if issue.get(
                        "dataset_rule" ):

                        st.markdown(
                            "**Closest AccessGuru violation:** "
                            f"`{issue['dataset_rule']}`"
                        )

                        st.markdown(
                            f"Dataset similarity: "
                            f"`{issue.get('dataset_similarity', 0):.3f}`"
                        )

                with right:

                    st.markdown("**🔴 What exactly is wrong**")
                    st.write(issue.get("description", ""))
                    st.markdown("**Detected value / evidence**")
                    st.code(issue.get("detected_value", "Not available"), language="text")
                    st.markdown("**👁 Why this matters for accessibility**")
                    st.write(issue.get("why_it_matters", ""))

                    st.markdown(
                        "**Detected HTML**"
                    )

                    st.code(
                        html.unescape(
                            issue["element"]
                        ),
                        language="html",
                    )

                st.markdown("**🟢 Recommended fix for THIS finding**")
                st.info(issue.get("specific_fix", issue.get("description", "")))

                st.markdown("**Recommended fix / AccessGuru evidence**")

                dataset_fix = (
                    issue.get(
                        "dataset_fix",
                        "",
                    )
                    or ""
                )

                # Avoid showing huge generated text blocks
                # when the dataset contains them.
                if (
                    dataset_fix
                    and dataset_fix.lower() not in [
                        "nan",
                        "none",
                    ]
                ):

                    st.info(
                        "Recommendation retrieved from the "
                        "closest matching AccessGuru examples."
                    )

                    st.markdown(
                        dataset_fix[:5000]
                    )

                # Practical generic code examples
                rule = issue["rule"]

                if rule == "image-alt":

                    st.code(
                        """<!-- Informative image -->
<img src="doctor.jpg"
     alt="Doctor speaking with a patient">

<!-- Decorative image -->
<img src="decoration.svg"
     alt="">""",
                        language="html",
                    )

                elif rule == "button-name":

                    st.code(
                        """<!-- Good -->
<button aria-label="Close menu">
    <svg aria-hidden="true">...</svg>
</button>""",
                        language="html",
                    )

                elif rule == "label":

                    st.code(
                        """<label for="email">
    Email address
</label>

<input
    id="email"
    type="email"
    name="email">""",
                        language="html",
                    )

                elif rule == "html-has-lang":

                    st.code(
                        """<html lang="en">""",
                        language="html",
                    )

                elif rule == "frame-title":

                    st.code(
                        """<iframe
    src="..."
    title="Product demonstration video">
</iframe>""",
                        language="html",
                    )

                elif rule == "video-caption":

                    st.code(
                        """<video controls>
    <source src="video.mp4"
            type="video/mp4">

    <track
        kind="captions"
        src="captions.vtt"
        srclang="en"
        label="English">
</video>""",
                        language="html",
                    )

                elif rule == "heading-order":

                    st.code(
                        """<h1>Products</h1>
<h2>Computers</h2>
<h3>Laptops</h3>""",
                        language="html",
                    )

                elif rule == "duplicate-id":

                    st.code(
                        """<!-- IDs must be unique -->
<input id="email">
<input id="phone">""",
                        language="html",
                    )

                elif rule == "aria-hidden-focus":

                    st.code(
                        """<!-- Do not make aria-hidden
     content focusable -->

<div aria-hidden="true">
    <span>Decorative content</span>
</div>""",
                        language="html",
                    )

                elif rule == "contrast":

                    st.code(
                        """/* Too low */
color: #777;
background: #fff;

/* Better: choose a darker foreground */
color: #333;
background: #fff;""",
                        language="css",
                    )

    # --------------------------------------------------------
    # Download report
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "## 📄 Export"
    )

    report_bytes = result["report"].encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download Accessibility Report",
        data=report_bytes,
        file_name="accessify_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Methodology
    # --------------------------------------------------------

    with st.expander(
        "🧠 How Accessify works"
    ):

        st.markdown(
            """
            **Step 1 — Web analysis**

            The application downloads the submitted webpage and
            parses its HTML.

            **Step 2 — Accessibility checks**

            It checks practical accessibility signals such as
            alternative text, button names, form labels, headings,
            landmarks, language, iframe titles, captions, duplicate
            IDs, ARIA usage and keyboard-related attributes.

            **Step 3 — AccessGuru ML**

            The AccessGuru dataset contains real-world accessibility
            violation examples. The application trains a lightweight
            TF-IDF + Ridge model to estimate violation impact from
            violation descriptions and HTML.

            **Step 4 — Semantic retrieval**

            Each detected issue is compared with the AccessGuru
            dataset using TF-IDF cosine similarity. The closest
            known violation is used to provide additional context
            and a dataset-grounded remediation.

            **Step 5 — Screening score**

            The detected issues and their estimated severity are
            combined into an interpretable 0–100 screening score.

            **Step 6 — Developer action**

            Instead of only saying "your website failed", the
            application shows the offending HTML and practical code
            examples for fixing it.
            """
        )

    st.caption(
        "Accessify • SDG 10: Reduced Inequalities • "
        "Automated accessibility screening"
    )
