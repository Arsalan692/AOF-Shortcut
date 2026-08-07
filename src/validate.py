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
    (r"\bcnic\b|\bid\s*no\b|\bid\s*number\b|\bsnic\b|\bnicop\b", "cnic"),
    (r"passport", "passport"),
    (r"\bntn\b", "ntn"),
    (r"e-?mail", "email"),
    # a mobile box must hold a mobile: "0221234567" is a perfectly valid Hyderabad
    # landline and still the wrong answer here, which only a separate type can say
    (r"mobile|\bcell\b|cellular", "mobile"),
    (r"landline|tel\.?\s*no|telephone|phone", "phone"),
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

NUMERIC = {"date", "cnic", "ntn", "phone", "mobile", "amount", "integer",
           "postcode", "digits", "account"}

# Which types the OCR reader should transcribe: character-exact identifiers carrying no
# language for a general VLM to lean on. Deliberately a different set from NUMERIC, which
# governs digit repair - a passport or IBAN must never have its letters "repaired" into
# digits, or PK24HABB... becomes PK248488...
OCR_TYPES = NUMERIC | {"iban", "passport"}


# Parentheticals on this form qualify a field, they do not name its contents:
# "First Name (As per CNIC/ID)" is a name, not a CNIC, and "Date of Issuance (CNIC)"
# is a date. Strip them before matching, or every name field types as cnic.
QUALIFIER_RE = re.compile(r"\s*\([^)]*\)")

# Labels that ask for a description, not a figure. Tested before LABEL_TYPES because the
# money and duration words sit *inside* them: "Other Source of Income (Please specify)"
# holds "None", "Line of Business/Industry/Source of Income" holds a trade, and
# "...the nature and expected months of high turnover" holds "Festival related; May and
# June". Getting this wrong is no longer just a failed format check - the type decides
# which model reads the field, so an amount-typed description goes to the OCR reader,
# which truncated "20 years" to "20 yea" in Total previous Work Experience.
DESCRIPTIVE_RE = re.compile(
    r"please specify|\(specify\)|- other|\bnature\b|industry|line of business|"
    r"source of income|source of wealth|\bdetails\b|description|work experience")


def infer_type(field: dict) -> str:
    """field is a crop-index record (label, kind, grid_hint, cells)."""
    hint = (field.get("grid_hint") or "").upper()
    if hint and set(hint) <= set("DMY") and len(hint) >= 6:
        return "date"
    label = QUALIFIER_RE.sub(" ", field.get("label") or "").lower()
    # the parenthetical is stripped above, so "(Please specify)" is matched on the raw label
    if DESCRIPTIVE_RE.search(label) or DESCRIPTIVE_RE.search((field.get("label") or "").lower()):
        return "text"
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

# ------------------------------------------------- Pakistani formats, as issued
# These are the real national formats, not patterns inferred from any sample form. They
# earn their place by catching values that are well-formed in general but impossible here:
# a misread landline came back as 11 digits on the 091 (Peshawar) area code, which only
# issues 7-digit subscriber numbers, so the value is provably wrong even though it looks
# like a phone number.

CNIC_RE = re.compile(r"^(\d{5})-?(\d{7})-?(\d)$")
# First digit of a CNIC is the province/region that issued it. 0 is not issued.
CNIC_REGION = {"1": "Khyber Pakhtunkhwa", "2": "FATA", "3": "Punjab", "4": "Sindh",
               "5": "Balochistan", "6": "Islamabad", "7": "Gilgit-Baltistan",
               "8": "Azad Kashmir"}

# Mobile: 03 + 9 more digits = 11 total. Operator ranges are 030x-034x plus SCO 0355,
# so the third digit is always 0-5. Deliberately no tighter: an allowlist of live
# operator prefixes would reject valid numbers as ranges are reassigned.
PK_MOBILE_RE = re.compile(r"^03[0-5]\d{8}$")

# Landline: area code (including its leading 0) + subscriber number. The two biggest
# cities issue 8-digit subscriber numbers, everywhere else issues 7, so total length is
# tied to the area code rather than being a free range.
PK_AREA_8DIGIT = {"021", "042"}                       # Karachi, Lahore
PK_AREA_CODES = {
    "021": "Karachi", "022": "Hyderabad", "025": "Dadu", "028": "Nawabshah",
    "040": "Sahiwal", "041": "Faisalabad", "042": "Lahore", "044": "Okara",
    "045": "Kasur", "046": "Toba Tek Singh", "047": "Jhang", "048": "Sargodha",
    "049": "Sheikhupura", "051": "Islamabad/Rawalpindi", "052": "Sialkot",
    "053": "Gujrat", "055": "Gujranwala", "056": "Sheikhupura", "057": "Attock",
    "061": "Multan", "062": "Bahawalpur", "063": "Vehari", "064": "Layyah",
    "065": "Muzaffargarh", "066": "Dera Ghazi Khan", "067": "Khanewal",
    "068": "Rahim Yar Khan", "071": "Sukkur", "074": "Larkana", "081": "Quetta",
    "091": "Peshawar", "092": "Mardan", "094": "Kohat", "095": "Bannu",
    "0992": "Abbottabad", "0997": "Haripur", "0946": "Swat",
}


def classify_pk_phone(digits: str) -> tuple[bool, str, str]:
    """-> (ok, message, canonical). digits is the value stripped to digits only.

    The canonical form keeps the dash Pakistan writes the number with - 0320-5120612,
    021-36102837 - rather than flattening to bare digits. It is how the number appears on
    the form and how a person expects to read it back.
    """
    if digits.startswith("03"):
        if PK_MOBILE_RE.match(digits):
            return True, "", f"{digits[:4]}-{digits[4:]}"
        if len(digits) != 11:
            return False, f"a Pakistani mobile is 11 digits as 03XX-XXXXXXX (got {len(digits)})", digits
        # right length, impossible network: operators hold 0300-0349 and SCO holds 0355
        return False, (f"{digits[:4]} is not a Pakistani mobile prefix "
                       "(they run 0300-0349, plus 0355)"), digits
    if not digits.startswith("0"):
        return False, "a Pakistani number starts 0 (03 mobile, or a city code such as 021)", digits
    for n in (4, 3):                                   # 0992 before 099
        code = digits[:n]
        if code in PK_AREA_CODES:
            want = 8 if code in PK_AREA_8DIGIT else 7
            got = len(digits) - n
            if got == want:
                return True, "", f"{code}-{digits[n:]}"
            return False, (f"{code} ({PK_AREA_CODES[code]}) issues {want}-digit numbers, "
                           f"so expected {n + want} digits, got {len(digits)}"), digits
    # unrecognised area code: do not reject, the list is not exhaustive
    if 10 <= len(digits) <= 11:
        return True, "", f"{digits[:3]}-{digits[3:]}"
    return False, "expected 10-11 digits for a landline, or 11 for a mobile", digits


# IBAN: PK + 2 check digits + 4-letter bank code + 16 alphanumeric = 24 characters.
IBAN_RE = re.compile(r"^PK\d{2}([A-Z]{4})([A-Z0-9]{16})$")
# This tool reads HBL's own account opening form, and its single IBAN field sits in the
# "For Bank Use Only" band - it is the HBL account being opened, so the bank code must be
# HBL's. A structurally perfect IBAN carrying anyone else's code is a misread here, not a
# different bank's account. (The other banks a customer names on page 4 are free text in
# "Name of the Bank/DFI" and are not validated against anything.)
HBL_BANK_CODE = "HABB"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
# Machine-readable Pakistani passport: 2 letters then 7 digits.
PK_PASSPORT_RE = re.compile(r"^[A-Z]{2}\d{7}$")


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
            norm = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if m.group(1)[0] not in CNIC_REGION:
                # 0 is not an issuing region, so this cannot be a real CNIC
                return False, norm, f"CNIC starts with {m.group(1)[0]}, not an issuing region"
            return True, norm, ""
        return False, v, "expected 13 digits as XXXXX-XXXXXXX-X"
    if ftype == "iban":
        s = re.sub(r"[\s-]", "", v).upper()
        m = IBAN_RE.match(s)
        if m:
            bank = m.group(1)
            if bank != HBL_BANK_CODE:
                return False, s, (f"bank code is {bank}, but an HBL IBAN reads "
                                  f"{HBL_BANK_CODE}")
            return True, s, ""
        return False, v, "expected PK + 2 digits + 4 letters + 16 alphanumerics (24)"
    if ftype == "ntn":
        s = re.sub(r"\D", "", v)
        # an FBR NTN is 7 digits, sometimes written with its check digit as 1234567-8;
        # an individual may instead give their 13-digit CNIC, which is also their NTN
        if len(s) in (7, 8, 13):
            return True, s, ""
        return False, v, "expected 7 digits (NTN), 8 with a check digit, or 13 (CNIC)"
    if ftype == "email":
        s = re.sub(r"\s", "", v)
        if EMAIL_RE.match(s):
            return True, s, ""
        return False, v, "not a valid e-mail address"
    if ftype == "mobile":
        s = re.sub(r"[^\d+]", "", v)
        s = re.sub(r"^(?:\+?92)", "0", s)
        ok, msg, canon = classify_pk_phone(s)
        if ok and canon.replace("-", "").startswith("03"):
            return True, canon, ""
        return False, v, msg or ("a Pakistani mobile is 11 digits starting 03 "
                                 f"(got {len(s)} digits starting {s[:3]})")
    if ftype == "phone":
        s = re.sub(r"[^\d+]", "", v)
        s = re.sub(r"^(?:\+?92)", "0", s)          # +92 3xx... is the same number as 03xx...
        ok, msg, canon = classify_pk_phone(s)
        return (True, canon, "") if ok else (False, v, msg)
    if ftype == "postcode":
        s = re.sub(r"\D", "", v)
        if len(s) == 5:                            # Pakistan Post uses 5-digit codes
            return True, s, ""
        return False, v, f"a Pakistani postal code is 5 digits (got {len(s)})"
    if ftype == "passport":
        s = re.sub(r"[\s-]", "", v).upper()
        if PK_PASSPORT_RE.match(s):
            return True, s, ""
        if re.fullmatch(r"[A-Z0-9]{7,9}", s):
            return True, s, "not the usual 2 letters + 7 digits for a Pakistani passport"
        return False, v, "expected 2 letters then 7 digits"
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
def cross_check(values: dict[str, dict], ticks: dict[str, str] | None = None) -> list[str]:
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

    # The last digit of a CNIC is the holder's gender: odd male, even female. The form asks
    # for gender separately, so the two must agree - a real consistency check on a value no
    # format rule can question, because a wrong digit still gives a well-formed CNIC.
    ticks = ticks or {}
    cnic = get(lambda f: "id_no" in f or "cnic" in f)
    gender = (ticks.get("Gender") or "").strip().lower()
    if cnic and gender in ("male", "female"):
        last = re.sub(r"\D", "", cnic)[-1:]
        if last.isdigit():
            implied = "male" if int(last) % 2 else "female"
            if implied != gender:
                out.append(f"CNIC ends in {last}, which means {implied}, "
                           f"but Gender is ticked {gender.title()}")

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
