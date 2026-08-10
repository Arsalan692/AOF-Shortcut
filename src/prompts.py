"""
Everything the vision model is told about what it is looking at.

The model reads a whole page in one call, so it never has to work out *which* field it is
reading - the page image carries a numbered marker on every answer box, and this module
produces the matching numbered list. What each entry has to say is the real work: told
only "Mobile Number", a reader will happily return ten digits or twelve; told "exactly 11
digits starting 03, real prefixes are 0300-0349 plus 0355", it has something to count its
own answer against.

The formats below are the Pakistani ones as issued, and the bank-specific ones are HBL's.
They are not inferred from any sample form - a filled form is dummy data and may well
contain values that break them, which is exactly what the reader must not copy from.

A rule these instructions must keep: **a spec describes the shape of a value, and never
gives a literal example that could itself be a valid answer to that field.** Written the
other way round it teaches the model to guess. This was not hypothetical - an early version
of the postcode spec offered "(74000 Karachi, 54000 Lahore, 44000 Islamabad)" as
illustration, and a 3B reader handed back 54000 for a box that plainly read 12345. Worse,
the first draft of these specs was written while looking at a filled sample form, so its
answers - a branch name, a district, a house number, an officer's initials - had leaked in
as "examples" and would have been fed back on every real form thereafter.

Enumerating a closed set is not the same thing and is fine: "either Clean or Secured", or
the list of relationship words, IS the answer space rather than a sample drawn from it.

Three sources feed a field's instruction, most specific first:

  FIELD_SPECS   by field id, for the fields whose printed label does not describe them
                (page 3's two bare "PKR" boxes, page 9's run-together signature labels)
  LABEL_SPECS   by label wording, for families the type system does not separate
                (Relationship, Nature (Clean/Secured), Name of the Bank/DFI)
  type_spec()   by validate.infer_type(), which is the same typing that decides
                validation - so what the model is asked for and what it is checked
                against can never drift apart
"""
from __future__ import annotations

import re

from validate import infer_type

# ------------------------------------------------------------------ constants
# HBL's own bank code inside a Pakistani IBAN. Pre-printed on the form, because the only
# two IBAN boxes on it (page 1's account, page 3's direct-debit account) are HBL accounts.
HBL_BANK_CODE = "HABB"
MAX_CARD_NAME = 19          # characters embossable on an HBL card, spaces included

# What the reader should answer when a box holds no handwriting at all.
EMPTY_TOKEN = "EMPTY"
# ...and when it holds a signature scribble rather than letters. Kept distinct from EMPTY:
# "nobody signed here" and "somebody signed here illegibly" are different facts, and only
# the first one means the form is incomplete.
SIGNATURE_TOKEN = "SIGNATURE"


def cells_note(rec: dict, expected: int | None = None) -> str:
    """"...across N printed boxes", but only when N agrees with the format.

    The character grids are harvested from the template, so the count is normally exact
    and worth quoting - a CNIC grid is 15 boxes for 13 digits and 2 dashes, which lets the
    reader check its answer against something it can count. It is not always exact:
    page 3's supplementary-card Mobile No. harvests as 9 boxes for an 11-digit number,
    because part of the grid merges with its neighbour. Quoting that would tell the reader
    to drop two digits, so a count that contradicts the format is dropped instead.
    """
    n = rec.get("cells") or 0
    if not n or (expected is not None and n != expected):
        return ""
    return f", one character per box across {n} printed boxes"


# ------------------------------------------------------------------ type specs
# The format for each type, said once. A page repeats its types heavily - page 1 alone has
# six dates and five person-names - and on CPU the prompt is not free: measured on that
# page, prefill was 76% of the call and the prompt text about half of the prefill. So a
# type used more than once on a page is hoisted into a glossary and referenced by tag,
# which says exactly the same thing without paying for it six times.
TYPE_BASE: dict[str, str] = {
    "date": ("a date as 8 digits in DDMMYYYY order - day 01-31, month 01-12, then a 4-digit "
             "year. The boxes carry faint pre-printed 'D D M M Y Y Y Y' letters; that is "
             "printing, not the answer. Where the date is written on a ruled line instead "
             "of in boxes the writer may use a single digit for the day or the month - "
             "'15-5-2024' is the 15th of May, so it is 15052024. Pad it with a leading "
             "zero; NEVER substitute a different number to make up the count"),
    "cnic": ("a Pakistani CNIC: exactly 13 digits written XXXXX-XXXXXXX-X. The first digit "
             "is the issuing province and is 1-8, never 0 (1 Khyber Pakhtunkhwa, 2 FATA, "
             "3 Punjab, 4 Sindh, 5 Balochistan, 6 Islamabad, 7 Gilgit-Baltistan, "
             "8 Azad Kashmir). Keep both dashes"),
    "iban": (f"a Pakistani IBAN for an HBL account: exactly 24 characters - PK, 2 check "
             f"digits, the 4-letter bank code {HBL_BANK_CODE}, then 16 more digits/letters. "
             f"'PK', '{HBL_BANK_CODE}' and the '00' after it are pre-printed on the form - "
             f"include them in the answer and read the handwritten rest"),
    "mobile": ("a Pakistani mobile number: exactly 11 digits starting 03, written "
               "03XX-XXXXXXX. Live prefixes run 0300-0349, plus 0355 for SCO. Count the "
               "digits"),
    "phone": ("a Pakistani landline: the city code (starting 0) then the subscriber number. "
              "Karachi 021 and Lahore 042 have 8-digit subscriber numbers (11 digits in "
              "total); every other city code has 7 (10 in total)"),
    "email": ("an e-mail address - one @ and a dot in the domain. Lower case unless the "
              "writer used capitals"),
    "passport": ("a Pakistani machine-readable passport number: 2 capital letters followed "
                 "by 7 digits"),
    "ntn": ("an FBR National Tax Number: 7 digits, or 8 if the writer added the check digit. "
            "An individual may instead write their 13-digit CNIC, which is also valid here"),
    "postcode": "a Pakistan Post code: exactly 5 digits",
    "amount": ("an amount in Pakistani Rupees: digits, possibly with thousands commas and a "
               "trailing /- . Copy the digits as written; do not round and do not add a "
               "currency word"),
    "integer": ("a whole number, digits only. Any printed unit beside the box ('years', "
                "'months') is form printing, not part of the answer"),
    "digits": "digits only, no letters",
    "account": "digits only, no letters",
    "name": ("a Pakistani person's or company's name: letters, spaces and possibly an "
             "initial with a full stop. No digits"),
    "text": "free text - copy exactly what is written",
}

# The cell count each type's grid should have, where the format fixes it. Used to decide
# whether the harvested count is worth quoting (see cells_note).
TYPE_CELLS: dict[str, int | None] = {
    "date": 8, "cnic": 15, "iban": 24, "mobile": 12, "phone": 12, "postcode": 5,
}

TYPE_TAGS = {t: t.upper() for t in TYPE_BASE}


def type_spec(t: str, rec: dict) -> str:
    """The full instruction for a field of type `t`, cell count included."""
    return TYPE_BASE.get(t, TYPE_BASE["text"]) + cells_note(rec, TYPE_CELLS.get(t))


# ------------------------------------------------------------- label families
# Matched against the lower-cased label when no id override applies and the type system
# has nothing more specific to say. Ordered: first match wins.
LABEL_SPECS: list[tuple[str, str]] = [
    (r"relationship",
     "a relationship word such as Wife, Husband, Son, Daughter, Father, Mother, "
     "Brother, Sister, Uncle, Cousin, Friend or Colleague"),
    (r"nature \(clean/secured\)",
     "either the word Clean or the word Secured"),
    (r"name of the bank/dfi",
     "the name of a Pakistani bank or DFI, as the customer wrote it"),
    (r"^sr\. no\.",
     "the row number: a single digit"),
    (r"facility under process",
     "the kind of facility applied for"),
    (r"nationalit",
     "a nationality, written as an adjective rather than a country name"),
    (r"^country|country of (birth|residence|stay)",
     "a country name"),
    (r"city of birth|place of birth|^city$",
     "a Pakistani city name. Unfamiliar spellings are normal - copy the letters exactly "
     "and never substitute a better-known city that merely looks similar"),
    (r"area/district",
     "an area, town or district name - a neighbourhood, a named town, or a compass "
     "sector of a city"),
    (r"nearest landmark",
     "a nearby landmark used to find the address. Often a single word"),
    (r"house/appt|office no\./office name|street no\./name",
     "an address line mixing letters and digits, and possibly / or - . After Block, "
     "Sector or Phase either a DIGIT or a LETTER is possible - copy exactly which one "
     "is written"),
    (r"^address",
     "a full address line: house or plot number, street, block or area, then city. "
     "It mixes letters, digits, commas and slashes - copy them as written"),
    (r"designation",
     "a job title"),
    (r"department",
     "the name of a department inside a company"),
    (r"name of company/employer|insurance company|dealer name|seller name",
     "an organisation's name. Copy unfamiliar or invented company names exactly - "
     "do not normalise them into a more familiar word"),
    (r"business details|line of business|source of income industry|nature/type of business",
     "a short description of a trade or industry"),
    (r"branch name",
     "an HBL branch name, usually written in capitals"),
    (r"region name",
     "an HBL region"),
    (r"^promotion$|^category$",
     "a short code or word entered by the branch; copy it exactly, including N/A"),
    (r"colour",
     "a colour name"),
    (r"car make|car model|manufacturer",
     "a vehicle manufacturer, make or model"),
    (r"year of manufacture",
     "a 4-digit year"),
    (r"insurance rate|equity/security deposit",
     "a percentage - digits, possibly with a decimal point and a % sign"),
    (r"identity document type",
     "either the word CNIC or the word Passport"),
    (r"occupation",
     "an occupation or trade"),
]


# ------------------------------------------------------ per-field overrides
# For fields whose printed label does not describe what goes in them. Most are cases where
# the label sits in a different cell from the box (page 3's bare "PKR" boxes) or where two
# printed captions run together during the harvest (page 9's signature rows).
FIELD_SPECS: dict[str, str] = {
    # ---- page 1, For Bank Use Only
    "p1_gri_branch_code":
        "an HBL branch code: 4 digits, one per box",
    "p1_gri_customer_number":
        "the customer number: 6 digits, one per box",
    "p1_tex_p_a_no":
        "the P.A. (Personal Authority) number of the approving officer: digits only, "
        "usually 5",
    "p1_tex_p_a_no__2":
        "the P.A. (Personal Authority) number of the officer who verified the signature: "
        "digits only, usually 5",
    "p1_blo_approved_by":
        "the approving officer's initials or short signature. Transcribe the letters as "
        f"written, even if heavily abbreviated. If it is an unreadable scribble "
        f"with no legible letters, answer {SIGNATURE_TOKEN}",
    "p1_blo_signature_verified_by":
        "the verifying officer's initials or short signature. Transcribe the letters as "
        f"written. If no letters can be made out, answer {SIGNATURE_TOKEN}",
    # ---- page 1, Personal Information
    "p1_tex_title_of_account_as_per_cnic_id":
        "the account title: the customer's full name exactly as on the CNIC, normally in "
        "CAPITAL letters. Letters and spaces only",
    "p1_tex_title_other_specify":
        "a title or honorific written instead of Mr./Mrs./Ms./Dr.",
    "p1_tex_marital_status_other_specify":
        "a marital status written in words",
    "p1_tex_education_other_specify":
        "a qualification not offered by the tick boxes",
    "p1_tex_number_of_dependents":
        "the number of dependents: one or two digits",
    "p1_tex_if_yes_please_list_nationalities":
        "one or more nationalities, separated by commas. Often left blank or N/A",
    # ---- page 1, Next of Kin
    "p1_tex_name":
        "the next of kin's full name: letters and spaces only",
    "p1_tex_telephone_number":
        "the next of kin's phone number - either a Pakistani mobile (11 digits starting "
        "03, as 03XX-XXXXXXX) or a landline (city code then 7-8 digits). Count the digits",
    # ---- page 2
    "p2_tex_customer_type_other_specify":
        "a customer type not offered by the tick boxes",
    "p2_tex_profession_other_specify":
        "a profession not offered by the tick boxes",
    "p2_tex_nature_of_business_other_specify":
        "a line of business not offered by the tick boxes",
    "p2_tex_source_of_income_other_specify":
        "an income source not offered by the tick boxes",
    "p2_tex_other_source_of_wealth_please_specify":
        "a source of wealth not offered by the tick boxes, or None/N/A. This is a "
        "description, not an amount",
    "p2_tex_please_specify_designation":
        "the customer's designation or job title",
    "p2_tex_total_previous_work_experience":
        "the number of years of previous work experience: digits only. The printed word "
        "'years' after the box is form printing, not part of the answer",
    "p2_tex_other_source_of_income_please_specify":
        "a short description of another income source, or None/N/A. This is a "
        "description, not an amount",
    "p2_tex_if_yes_please_specify_the_nature_and_expected_months_of_high_turnover":
        "a sentence naming the reason for the seasonal activity and the months it "
        "falls in",
    "p2_tex_amount_pkr":
        "the expected monthly CREDIT turnover in Pakistani Rupees: digits, possibly with "
        "thousands commas",
    "p2_tex_amount_pkr__2":
        "the expected monthly DEBIT turnover in Pakistani Rupees: digits, possibly with "
        "thousands commas",
    "p2_tex_no_of_transactions":
        "the expected number of credit transactions per month: digits only, usually 1-2 "
        "digits",
    "p2_tex_no_of_transactions__2":
        "the expected number of debit transactions per month: digits only, usually 1-2 "
        "digits",
    # ---- page 3
    "p3_tex_identity_document_type_cnic_passport":
        "either the word CNIC or the word Passport",
    "p3_tex_id_number":
        "the financial supporter's identity number. If the document type says CNIC this "
        "is 13 digits as XXXXX-XXXXXXX-X with the first digit 1-8; if it says Passport it "
        "is 2 capital letters and 7 digits",
    "p3_gri_cif_number_if_available":
        "the customer's CIF number: 6 digits, one per box",
    "p3_tex_pkr":
        "the Equity/Security Deposit amount in Pakistani Rupees: digits, possibly with "
        "thousands commas",
    "p3_tex_pkr__2":
        "the Residual Value / Deferred Amount in Pakistani Rupees: digits, possibly with "
        "thousands commas",
    "p3_gri_name_to_appear_on_hbl_credit_card":
        f"the name to be embossed on the HBL credit card: CAPITAL letters and spaces "
        f"only, at most {MAX_CARD_NAME} characters, one letter per box. A blank box is a "
        "word gap",
    "p3_gri_name_to_appear_on_supplementary_card":
        f"the name to be embossed on the supplementary card: CAPITAL letters and spaces "
        f"only, at most {MAX_CARD_NAME} characters, one letter per box",
    "p3_gri_basic_card_no":
        "the basic HBL card number: 16 digits, one per box",
    "p3_gri_card":
        "the supplementary cardholder's CNIC: exactly 13 digits as XXXXX-XXXXXXX-X "
        "across 15 boxes (each dash takes a box). The first digit is 1-8, never 0",
    "p3_gri_landline_no":
        "the supplementary cardholder's landline: the city code (starting 0) then 7 "
        "digits, 8 for Karachi 021 and Lahore 042",
    "p3_gri_mobile_no":
        "the supplementary cardholder's mobile: exactly 11 digits starting 03, as "
        "03XX-XXXXXXX. Live prefixes run 0300-0349, plus 0355",
    "p3_gri_direct_debit_a_c_no_repayment_a_c_no":
        f"the HBL account the repayment is debited from, as a Pakistani IBAN: 24 "
        f"characters - PK, 2 check digits, {HBL_BANK_CODE}, then 16 more. 'PK', "
        f"'{HBL_BANK_CODE}' and the following '00' are pre-printed; include them",
    # ---- page 4
    "p4_gri_reference_1_name":
        "the first reference's full name in CAPITAL letters, one letter per box. "
        "A blank box is a gap between words",
    "p4_gri_reference_2_name":
        "the second reference's full name in CAPITAL letters, one letter per box. "
        "A blank box is a gap between words",
    "p4_blo_name_of_applicant":
        "the applicant's own full name, handwritten in normal case rather than capitals. "
        "Letters and spaces only",
    "p4_blo_signature":
        f"the applicant's signature. If any letters are legible transcribe them, "
        f"otherwise answer {SIGNATURE_TOKEN}",
    # ---- page 9
    "p9_gri_primary_applicant_date_signature":
        "the date written beside the PRIMARY APPLICANT's signature: 8 digits DDMMYYYY",
    "p9_gri_supplementary_card_date_holder_s_signature":
        "the date written beside the SUPPLEMENTARY CARD HOLDER's signature: 8 digits "
        "DDMMYYYY",
    "p9_gri_co_borrower_1_signature_co_borrower_2_signature_date_car_loan":
        "the date written beside the CO-BORROWER signatures: 8 digits DDMMYYYY",
    "p9_gri_cif_number":
        "the CIF number entered by the branch: 6 digits, one per box",
    "p9_tex_bank_branch_code":
        "the HBL branch code: 4 digits",
    "p9_gri_date":
        "the date entered by the branch: 8 digits DDMMYYYY",
    "p9_tex_sc_rso_signature_sm_name":
        "the Sales Manager's (SM) name: letters and spaces only",
    "p9_tex_personnel_no_sales_code":
        "the SC/RSO's personnel number or sales code: digits only",
    "p9_tex_referrer_personnel_id_code":
        "the referrer's personnel ID or code: digits only",
    "p9_tex_employee_no":
        "the processor's employee number: digits only",
    "p9_tex_employee_no__2":
        "the sales manager's employee number: digits only",
    "p9_tex_attorney_no":
        "the officer's attorney number: digits only",
    "p9_tex_attorney_no__2":
        "the manager's attorney number: digits only",
    "p9_tex_p_no":
        "the officer's P. No. (personnel number): digits only",
    "p9_tex_p_no__2":
        "the manager's P. No. (personnel number): digits only",
    "p9_blo_sc_rso_signature":
        f"the SC/RSO's signature. Transcribe any legible letters, otherwise answer "
        f"{SIGNATURE_TOKEN}",
    "p9_blo_referrer_personnel_id_code_sign_employee_no":
        f"the processor's signature. Transcribe any legible letters, otherwise answer "
        f"{SIGNATURE_TOKEN}",
    "p9_blo_sign":
        f"the sales manager's signature. Transcribe any legible letters, otherwise "
        f"answer {SIGNATURE_TOKEN}",
    "p9_blo_signature":
        f"the officer's signature. Transcribe any legible letters, otherwise answer "
        f"{SIGNATURE_TOKEN}",
    "p9_blo_signature__2":
        f"the manager's signature. Transcribe any legible letters, otherwise answer "
        f"{SIGNATURE_TOKEN}",
}

# A signature block is a scribble, not a value: a general instruction covers any block
# field that has no entry of its own above.
BLOCK_SPEC = (f"a handwritten name or signature. Transcribe the letters if any can be "
              f"read, otherwise answer {SIGNATURE_TOKEN}")


# The two types that say almost nothing about content. Everything else is a real format -
# a date is 8 digits whatever the label around it says - so for those the format wins and
# the label rules are never consulted.
WEAK_TYPES = {"text", "name"}


def spec_source(rec: dict) -> tuple[str | None, str]:
    """(shared type key, full instruction) for one field.

    The type key is None whenever the instruction is specific to this field, which is what
    tells the page prompt it cannot be shared into the glossary.
    """
    fid = rec.get("field") or rec.get("id") or ""
    if fid in FIELD_SPECS:
        return None, FIELD_SPECS[fid]

    label = (rec.get("label") or "").lower()
    t = infer_type(rec)
    if t not in WEAK_TYPES:
        return t, type_spec(t, rec)
    # "name" is weak because the form's own labels put the word in places that are not a
    # person: Branch Name, Dealer Name, Car Make/Model, Insurance Company. Left to the type
    # alone they would all be described as "a Pakistani person's name, no digits".
    for pat, text in LABEL_SPECS:
        if re.search(pat, label):
            return None, text
    if rec.get("kind") == "block":
        return None, BLOCK_SPEC
    return t, type_spec(t, rec)


def spec(rec: dict) -> str:
    """The full instruction for one field. rec is a field-index record."""
    return spec_source(rec)[1]


# --------------------------------------------------------------- page prompt
HOW_TO_READ = f"""HOW TO READ THIS PAGE
- Every box you must read is outlined in blue and carries a red number tag placed just to
  its left or just above it. The tag belongs to the box it touches.
- Handwriting on these forms regularly overflows its box, crossing into the rows above and
  below. An answer belongs to the box its writing STARTS in.
- Ignore everything printed on the form: the English labels, the Urdu translations beside
  them, the box borders, and the faint pre-printed placeholder letters inside character
  grids (date grids are printed "D D M M Y Y Y Y", the IBAN grid is printed "PK  HABB 00").
  A printed unit beside a box, such as "years" after a handwritten "20", is not part of
  the answer.
- Tick boxes and checkboxes are not in the list below. Never report them.
- These are Pakistani names, places, companies and numbers, filled in by hand in Pakistan.
  Transcribe exactly what the pen shows, letter by letter and digit by digit. Do NOT turn
  an unusual word into a more common one - the unfamiliar reading is usually the correct
  one - and do not tidy spelling, spacing or capitalisation. If the pen wrote "Chaudry",
  the answer is "Chaudry" and not "Chaudhry"; if it wrote "Thang", do not answer "Jhang".
- Read every numbered box on its own. Do NOT reason about whether a field ought to apply to
  this customer - a Passport box on a Pakistani customer's form, or a supplementary-card box
  when no supplementary card was asked for, is still read if there is handwriting in it.
  Your job is what the pen put on the paper, not whether it makes sense.
- A box next to tick boxes, usually labelled "Other (specify)", is filled in ONLY when the
  writer ticked "Other" and then wrote in it. Never copy the wording of a ticked printed
  option into such a box - if the writer ticked a printed option, the "Other" box is
  "{EMPTY_TOKEN}".
- Never invent a value to fit the format. The format tells you what to LOOK for, not what
  to answer. If what is written breaks it, report what is written.
- If a numbered box holds no handwriting at all, answer "{EMPTY_TOKEN}". Check first: a box
  is only empty when there are no pen strokes inside its blue outline.
- If the writer put "N/A", "None" or a dash to mean "does not apply", answer exactly that.
"""


# ------------------------------------------------------------------ ink evidence
# Reading a whole page in one pass, the model skims: shown 45 boxes it answered EMPTY for a
# Passport box holding 7,961px of pen, and copied a ticked option's printed label into an
# "Other (specify)" box holding 186px of a neighbour's bleed. It is not guessing blindly in
# either case - it simply is not looking hard at one box out of 45.
#
# We already know the answer to "is there writing in this box", from the ink layer and the
# ownership partition, so the prompt says so. This is measurement handed to the model, not
# another instruction to be more careful.
INK_TRACE_PX = 500      # at or below this, ink in a box is a neighbour's overflow, not an
                        # answer: page 1's empty "Other (specify)" boxes hold 186/328/373px
                        # while its smallest real answer - a single digit - holds 827
INK_SURE_PX = 4000      # comfortably above any residue, so "this is filled" is safe to state
BASELINE_MULT = 3       # ...except where the template prints inside the box. A date grid
                        # carries faint "D D M M Y Y Y Y" glyphs that survive subtraction,
                        # and README measured empty (1342px) against filled (1358px) - not
                        # separable. Requiring the ink to beat the printed baseline several
                        # times over stands every date grid down, filled or not.


def ink_verdict(rec: dict) -> str:
    """"filled" | "blank" | "" - what the pixels say, where they are conclusive.

    Deliberately silent in the middle. The point of this signal is that it can be relied on,
    so it speaks only where the measurement is unambiguous, and the caller gets nothing
    rather than a guess everywhere else.
    """
    ink = rec.get("ink_px", -1)
    base = rec.get("baseline_px", 0) or 0
    if ink < 0:
        return ""
    if ink <= INK_TRACE_PX:
        return "blank"
    if ink >= INK_SURE_PX and ink >= BASELINE_MULT * base:
        return "filled"
    return ""


# Terse on purpose. These notes go on most fields of a page, and the prompt is the dominant
# cost of the call, so the wording is the shortest that still states the fact.
INK_NOTES = {
    "blank": " [NO INK: blank box, answer EMPTY unless a value is plainly written]",
    "filled": " [INK PRESENT: this box IS filled - read it, never answer EMPTY]",
}


def ink_note(rec: dict) -> str:
    """What the pixels say about whether this box was filled in, when they are sure."""
    return INK_NOTES.get(ink_verdict(rec), "")


def field_line(n: int, rec: dict, cont_of: int | None = None,
               has_cont: int | None = None, what: str | None = None) -> str:
    """One numbered entry in the prompt's field list."""
    what = spec(rec) if what is None else what
    if cont_of is not None:
        # The long answers on this form wrap, and the second line is where a whole-page read
        # loses text: having answered field N with the first line, the model treats the
        # answer as finished and reports the continuation box as empty even though there is
        # obvious handwriting in it. So this says outright that the box is probably filled.
        return (f'{n}. "{rec["label"]}" - the SECOND line of field {cont_of}, which wrapped. '
                f"Look at the handwriting BELOW field {cont_of} and report that lower line "
                f"here, in full. It is a real answer, not a repeat: the address in field "
                f"{cont_of} is incomplete without it. Only answer \"{EMPTY_TOKEN}\" if there "
                f"is genuinely no second line of writing{ink_note(rec)}")
    tail = ""
    if has_cont is not None:
        tail = (f". This answer wraps onto a second line - report only the text on THIS line "
                f"here, and report the rest as field {has_cont}, which is not optional")
    return f'{n}. "{rec["label"]}" - expect {what}{tail}{ink_note(rec)}'


def page_prompt(page: int, items: list[dict], total_pages: int = 9) -> str:
    """The whole instruction for one page: how to read it, then every field in order."""
    pos = {it["field"]: i for i, it in enumerate(items, 1)}
    sources = [spec_source(it) for it in items]

    # A type used more than once on this page is worth defining once and referring to.
    counts: dict[str, int] = {}
    for t, _ in sources:
        if t:
            counts[t] = counts.get(t, 0) + 1
    shared = {t for t, c in counts.items() if c > 1}
    glossary = "".join(f"\n  [{TYPE_TAGS[t]}] {TYPE_BASE[t]}" for t in sorted(shared))
    glossary = (f"""
FORMATS USED BELOW. Each is defined once here and referred to by its tag:{glossary}
""" if shared else "")

    lines: list[str] = []
    heading = None
    for n, (it, (t, text)) in enumerate(zip(items, sources), 1):
        group = " > ".join(x for x in (it.get("section"), it.get("table")) if x)
        if group != heading:
            heading = group
            lines.append(f"\n=== {group or 'Other fields on this page'} ===")
        what = (f"[{TYPE_TAGS[t]}]{cells_note(it, TYPE_CELLS.get(t))}"
                if t in shared else text)
        parent = it.get("continuation_of") or ""
        lines.append(field_line(
            n, it,
            cont_of=pos.get(parent) if parent else None,
            has_cont=next((pos[c["field"]] for c in items
                           if c.get("continuation_of") == it["field"]), None),
            what=what,
        ))

    return f"""This is page {page} of {total_pages} of HBL's "Consumer Products Application Form /
Customer Information Form" - a Pakistani bank account opening form, filled in by hand and
then scanned. You are transcribing it for a bank reviewer.

There are {len(items)} numbered boxes to read on this page.

{HOW_TO_READ}{glossary}
READ THE FIELDS IN THIS ORDER. The section headings are context, not fields:
{chr(10).join(lines)}

Reply with JSON only: one entry for every number from 1 to {len(items)}, mapping the
number as a string to the value you read in that numbered box."""


# ------------------------------------------------------------- re-read prompt
def retry_prompt(items: list[dict], reasons: dict[str, str]) -> str:
    """The instruction for the second pass, which re-reads only the fields that failed.

    Here the fields have been cut out and stacked, so there is no page context left - and
    that is the point: each strip is large, holds one answer, and comes with the reason its
    first reading was rejected.
    """
    lines = []
    for n, it in enumerate(items, 1):
        why = reasons.get(it["field"], "")
        lines.append(f'{n}. "{it["label"]}" - expect {spec(it)}'
                     + (f'. The first reading was rejected: {why}' if why else ""))
    return f"""This image holds {len(items)} fields cut out of one scanned Pakistani bank form.
They are stacked vertically, separated by black lines, and numbered on the left.

Each of these was read once already and the answer did not fit the field's format, so read
them again, carefully and from the pen strokes alone.

The numbered fields are:
{chr(10).join(lines)}

- Read ONLY the handwriting in each strip. Ignore printed labels, box borders, Urdu text
  and the faint pre-printed placeholder letters in character grids.
- Count the characters before answering: if the field says 11 digits, return exactly 11.
- Do not answer with a plausible-looking value. If the writing genuinely breaks the
  expected format, report what is actually written.
- If a strip holds no handwriting, answer "{EMPTY_TOKEN}".

Reply with JSON only: one entry for every number from 1 to {len(items)}."""
