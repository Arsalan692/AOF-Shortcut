"""
Phase D - decide what each field should look like, then check and repair what was read.

This is where accuracy on the structured fields is won. A vision model reading cursive
will occasionally return "0300-402301O" or "2306202G"; the template already tells us
that field is 8 date digits or an 11-digit mobile number, so most such errors are
detectable and many are repairable without asking anything twice.

Repairs are deliberately one-directional and conservative: a substitution is accepted
only if it turns an invalid value into a valid one. Nothing is "corrected" into a
different valid value, and every change is reported.
"""
from __future__ import annotations

import calendar
import re

# --------------------------------------------------------------------- typing
# Ordered: the first pattern that matches the field's label wins.
LABEL_TYPES: list[tuple[str, str]] = [
    (r"\biban\b", "iban"),
    (r"\bcnic\b|\bid\s*no\b|\bid\s*number\b|\bsnic\b", "cnic"),
    (r"\bntn\b", "ntn"),
    (r"e-?mail", "email"),
    (r"mobile|landline|tel\.?\s*no|telephone|phone", "phone"),
    (r"date\b|\bsince\b|year of manufacture", "date"),
    (r"amount|income|\bpkr\b|turnover|\blimit\b|price|value|deposit|salary|outstanding",
     "amount"),
    (r"no\.? of transactions|number of dependents|term of loan|work experience|"
     r"\bmonths\b|\byears\b", "integer"),
    (r"post/?zip|zip code", "postcode"),
    (r"branch code|customer number|p\.?a\.?\s*no|personnel id|\bcode\b|\bp\.?\s*no\b",
     "digits"),
    (r"a/?c no|account no|card no", "account"),
    # An address line legitimately holds digits and slashes - "H. No. 43/2" - so it must
    # be caught before the `name` rule below, which the form's own labels would otherwise
    # apply to "House/Appt. No./Appt. Name", "Street No./Name" and "Office No./Office
    # Name" and then reject every real value they hold. This sits after `email` on purpose:
    # "Email Address" contains "address" and has already been typed by then.
    (r"house|appt|apartment|\bflat\b|\bplot\b|street|\broad\b|address|building|"
     r"\bfloor\b|\bblock\b|sector|village|mohallah?|landmark|\btown\b|office no", "text"),
    (r"name", "name"),
]

NUMERIC = {"date", "cnic", "ntn", "phone", "amount", "integer", "postcode",
           "digits", "account"}


# Parentheticals on this form qualify a field, they do not name its contents:
# "First Name (As per CNIC/ID)" is a name, not a CNIC, and "Date of Issuance (CNIC)"
# is a date. Strip them before matching, or every name field types as cnic.
QUALIFIER_RE = re.compile(r"\s*\([^)]*\)")


def infer_type(field: dict) -> str:
    """field is a crop-index record (label, kind, grid_hint, cells)."""
    hint = (field.get("grid_hint") or "").upper()
    if hint and set(hint) <= set("DMY") and len(hint) >= 6:
        return "date"
    label = QUALIFIER_RE.sub(" ", field.get("label") or "").lower()
    for pat, t in LABEL_TYPES:
        if re.search(pat, label):
            return t
    return "text"


# ----------------------------------------------------------------- validation
# "EMPTY" is the sentinel the prompt asks for when there is no handwriting at all.
# The rest are things a customer actually writes to mean "does not apply" - both end up
# as a null value, but only the latter is evidence that the field really was filled in.
SENTINELS = {"", "empty"}
# A reader that traces the printed rule into its answer hands back "-Karachi" or
# "_Pakistan". The dash is the box, not a character the writer put there, so removing it
# is not a correction of content and is exempt from the invalid->valid rule below.
# Leading side only, and never the full stop: "Shabn." and "H. No." end in a real one.
LEAD_ARTEFACT = " _*|=–—-"
NONE_MARKERS = {"none", "n/a", "na", "-", "--", "nil", "null", "no"}
EMPTY_TOKENS = SENTINELS | NONE_MARKERS

CNIC_RE = re.compile(r"^(\d{5})-?(\d{7})-?(\d)$")
IBAN_RE = re.compile(r"^PK\d{2}[A-Z]{4}[A-Z0-9]{16}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?\d[\d\s-]{7,16}\d$")


def _valid_date8(s: str) -> bool:
    if not re.fullmatch(r"\d{8}", s):
        return False
    d, m, y = int(s[:2]), int(s[2:4]), int(s[4:])
    if not (1 <= m <= 12 and 1900 <= y <= 2100):
        return False
    return 1 <= d <= calendar.monthrange(y, m)[1]


def check(value: str, ftype: str) -> tuple[bool, str, str]:
    """-> (ok, normalised, message). Assumes value is already non-empty."""
    v = value.strip()
    if ftype == "date":
        digits = re.sub(r"\D", "", v)
        if _valid_date8(digits):
            return True, digits, ""
        return False, v, "not a valid DDMMYYYY date"
    if ftype == "cnic":
        m = CNIC_RE.match(re.sub(r"\s", "", v))
        if m:
            return True, f"{m.group(1)}-{m.group(2)}-{m.group(3)}", ""
        return False, v, "expected 13 digits as XXXXX-XXXXXXX-X"
    if ftype == "iban":
        s = re.sub(r"[\s-]", "", v).upper()
        if IBAN_RE.match(s):
            return True, s, ""
        return False, v, "expected PK + 2 digits + 4 letters + 16 alphanumerics (24)"
    if ftype == "ntn":
        s = re.sub(r"\D", "", v)
        if 6 <= len(s) <= 13:
            return True, s, ""
        return False, v, "expected 6-13 digits"
    if ftype == "email":
        s = re.sub(r"\s", "", v)
        if EMAIL_RE.match(s):
            return True, s, ""
        return False, v, "not a valid e-mail address"
    if ftype == "phone":
        s = re.sub(r"[^\d+-]", "", v)
        if PHONE_RE.match(s) and 9 <= len(re.sub(r"\D", "", s)) <= 15:
            return True, s, ""
        return False, v, "expected 9-15 digits"
    if ftype == "amount":
        s = re.sub(r"[,\s/]|(?:pkr)|(?:rs\.?)", "", v, flags=re.I).rstrip("-")
        if re.fullmatch(r"\d+(?:\.\d{1,2})?", s):
            return True, s, ""
        return False, v, "expected a number"
    if ftype in ("integer", "postcode", "digits", "account"):
        s = re.sub(r"[\s-]", "", v)
        if re.fullmatch(r"\d+", s):
            return True, s, ""
        return False, v, "expected digits only"
    if ftype == "name":
        if re.fullmatch(r"[A-Za-z][A-Za-z .'’/-]{0,60}", v):
            return True, re.sub(r"\s+", " ", v), ""
        return False, v, "unexpected characters for a name"
    return True, re.sub(r"\s+", " ", v), ""


# --------------------------------------------------------------------- repair
# Shapes a pen makes that a reader confuses, in the numeric direction only.
DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0",
                             "I": "1", "l": "1", "|": "1", "i": "1",
                             "S": "5", "s": "5", "B": "8", "Z": "2",
                             "G": "6", "b": "6", "T": "7", "A": "4"})


def repair(value: str, ftype: str) -> tuple[str, str]:
    """Try to make an invalid value valid. -> (value, note). Never changes a value
    that already validates, and never turns one valid value into another."""
    if ftype not in NUMERIC:
        return value, ""
    fixed = value.translate(DIGIT_FIXES)
    if fixed == value:
        return value, ""
    ok, norm, _ = check(fixed, ftype)
    if ok:
        return norm, f"repaired letter/digit confusion: {value!r} -> {norm!r}"
    return value, ""


def assess(value: str | None, field: dict) -> dict:
    """Full verdict for one read value against its field."""
    ftype = infer_type(field)
    raw = (value or "").strip()
    # readers sometimes bracket a value with underscores or dashes tracing the rule
    core = raw.strip(" _*:.–-")
    if core.lower() in EMPTY_TOKENS:
        # "the writer put N/A here" is a different fact from "nothing was read", and
        # only the latter is suspicious when the pixels say there is ink present
        explicit = core.lower() in NONE_MARKERS
        return {"type": ftype, "value": None, "valid": True, "empty": True,
                "explicit_none": explicit, "raw": raw,
                "note": f"written as {core!r}" if explicit else ""}

    # after the empty test, so a lone "-" still counts as the writer's "does not apply".
    # raw itself is left alone: it is what the model said, and the record reports it.
    body = raw.lstrip(LEAD_ARTEFACT)
    ok, norm, msg = check(body, ftype)
    note = ""
    if not ok:
        norm2, note = repair(body, ftype)
        if note:
            ok, norm, msg = True, norm2, ""
    return {"type": ftype, "value": norm, "valid": ok, "empty": False,
            "explicit_none": False, "note": note or msg, "raw": raw}


# ------------------------------------------------------------- cross-checks
def cross_check(values: dict[str, dict]) -> list[str]:
    """Consistency rules that span fields. values maps field id -> assess() result.

    Kept few and certain. The point is to catch a misread that happens to be
    individually well-formed - a date that validates but precedes the issue date, a
    name that validates but disagrees with the account title.
    """
    out = []

    def get(pred):
        for fid, v in values.items():
            if pred(fid) and not v.get("empty") and v.get("value"):
                return v["value"]
        return None

    def d8(s):
        return (int(s[4:]), int(s[2:4]), int(s[:2])) if s and len(s) == 8 else None

    issue = d8(get(lambda f: "date_of_issue" in f))
    expiry = d8(get(lambda f: "date_of_expiry" in f))
    dob = d8(get(lambda f: "date_of_birth" in f and "p1" in f))
    if issue and expiry and expiry <= issue:
        out.append("Date of Expiry is not after Date of Issue")
    if dob and issue and dob >= issue:
        out.append("Date of Birth is not before Date of Issue")
    if dob and dob[0] > 2015:
        out.append(f"Date of Birth year {dob[0]} looks implausible")

    title = get(lambda f: "title_of_account" in f)
    first = get(lambda f: "first_name" in f)
    last = get(lambda f: "last_name" in f)
    if title and first and last:
        t = re.sub(r"[^a-z]", "", title.lower())
        if re.sub(r"[^a-z]", "", first.lower()) not in t \
                and re.sub(r"[^a-z]", "", last.lower()) not in t:
            out.append(f"Title of Account {title!r} matches neither First nor Last name")
    return out


if __name__ == "__main__":
    CASES = [
        ("23062026", {"label": "CIF Opening Date", "grid_hint": "DDMMYYYY"}),
        ("2306202G", {"label": "CIF Opening Date", "grid_hint": "DDMMYYYY"}),
        ("32132026", {"label": "CIF Opening Date", "grid_hint": "DDMMYYYY"}),
        ("35810-0234568-5", {"label": "ID No.", "grid_hint": ""}),
        ("358100234568 5", {"label": "ID No.", "grid_hint": ""}),
        ("PK24HABB0033256018874102", {"label": "IBAN", "grid_hint": ""}),
        ("PK24HABB00332560188741", {"label": "IBAN", "grid_hint": ""}),
        ("0316-3264890", {"label": "Telephone Number", "grid_hint": ""}),
        ("ashraf24@gmail.com", {"label": "Email Address", "grid_hint": ""}),
        ("300,000/-", {"label": "Expected Monthly Income (PKR)", "grid_hint": ""}),
        ("Chaudry Waqar", {"label": "Father's Name", "grid_hint": ""}),
        ("N/A", {"label": "Husband's Name", "grid_hint": ""}),
        ("EMPTY", {"label": "Nationality", "grid_hint": ""}),
        ("l2345", {"label": "Post/Zip Code", "grid_hint": ""}),
    ]
    print(f"{'input':<28}{'type':<10}{'valid':<7}{'value':<26}note")
    for val, f in CASES:
        f.setdefault("kind", "text"); f.setdefault("cells", 0)
        r = assess(val, f)
        print(f"{val[:27]:<28}{r['type']:<10}{str(r['valid']):<7}"
              f"{str(r['value'])[:25]:<26}{r['note']}")
