import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_CSV = "data/cleaned_for_llm_391_participant_otree.csv"
BASE_OUTPUT_DIR = "data/prolific_data"

UBI_MESSAGES_PATH = os.path.join(BASE_OUTPUT_DIR, "UBI_messages.json")
PENALTY_MESSAGES_PATH = os.path.join(BASE_OUTPUT_DIR, "penalty_messages.json")
WEIGHT_LOSS_MESSAGES_PATH = os.path.join(BASE_OUTPUT_DIR, "weight_loss_messages.json")

# Prolific message_order is 0..5
PERMS = [
    (1, 2, 3),  # index 0
    (1, 3, 2),  # index 1
    (2, 1, 3),  # index 2
    (2, 3, 1),  # index 3
    (3, 1, 2),  # index 4
    (3, 2, 1),  # index 5
]

# =========================
# COUNTRY MAP (FULL LIST)
# =========================
COUNTRY_CHOICES: List[Tuple[str, str]] = [
    ("AF", "Afghanistan"),
    ("AL", "Albania"),
    ("DZ", "Algeria"),
    ("AS", "American Samoa"),
    ("AD", "Andorra"),
    ("AO", "Angola"),
    ("AI", "Anguilla"),
    ("AQ", "Antarctica"),
    ("AG", "Antigua and Barbuda"),
    ("AR", "Argentina"),
    ("AM", "Armenia"),
    ("AW", "Aruba"),
    ("AU", "Australia"),
    ("AT", "Austria"),
    ("AZ", "Azerbaijan"),
    ("BS", "Bahamas"),
    ("BH", "Bahrain"),
    ("BD", "Bangladesh"),
    ("BB", "Barbados"),
    ("BY", "Belarus"),
    ("BE", "Belgium"),
    ("BZ", "Belize"),
    ("BJ", "Benin"),
    ("BM", "Bermuda"),
    ("BT", "Bhutan"),
    ("BO", "Bolivia"),
    ("BA", "Bosnia and Herzegovina"),
    ("BW", "Botswana"),
    ("BV", "Bouvet Island"),
    ("BR", "Brazil"),
    ("IO", "British Indian Ocean Territory"),
    ("VG", "British Virgin Islands"),
    ("BN", "Brunei"),
    ("BG", "Bulgaria"),
    ("BF", "Burkina Faso"),
    ("BI", "Burundi"),
    ("KH", "Cambodia"),
    ("CM", "Cameroon"),
    ("CA", "Canada"),
    ("CV", "Cape Verde"),
    ("KY", "Cayman Islands"),
    ("CF", "Central African Republic"),
    ("EA", "Ceuta and Melilla"),
    ("TD", "Chad"),
    ("CL", "Chile"),
    ("CN", "China"),
    ("CX", "Christmas Island"),
    ("CC", "Cocos Islands"),
    ("CO", "Colombia"),
    ("KM", "Comoros"),
    ("CK", "Cook Islands"),
    ("CR", "Costa Rica"),
    ("HR", "Croatia"),
    ("CU", "Cuba"),
    ("CW", "Curaçao"),
    ("CY", "Cyprus"),
    ("CZ", "Czech Republic"),
    ("CD", "Democratic Republic of the Congo"),
    ("DK", "Denmark"),
    ("DJ", "Djibouti"),
    ("DM", "Dominica"),
    ("DO", "Dominican Republic"),
    ("EC", "Ecuador"),
    ("EG", "Egypt"),
    ("SV", "El Salvador"),
    ("GQ", "Equatorial Guinea"),
    ("ER", "Eritrea"),
    ("EE", "Estonia"),
    ("SZ", "Eswatini"),
    ("ET", "Ethiopia"),
    ("FK", "Falkland Islands"),
    ("FO", "Faroe Islands"),
    ("FJ", "Fiji"),
    ("FI", "Finland"),
    ("FR", "France"),
    ("GF", "French Guiana"),
    ("PF", "French Polynesia"),
    ("TF", "French Southern Territories"),
    ("GA", "Gabon"),
    ("GM", "Gambia"),
    ("GE", "Georgia"),
    ("DE", "Germany"),
    ("GH", "Ghana"),
    ("GI", "Gibraltar"),
    ("GR", "Greece"),
    ("GL", "Greenland"),
    ("GD", "Grenada"),
    ("GP", "Guadeloupe"),
    ("GU", "Guam"),
    ("GT", "Guatemala"),
    ("GG", "Guernsey"),
    ("GN", "Guinea"),
    ("GW", "Guinea-Bissau"),
    ("GY", "Guyana"),
    ("HT", "Haiti"),
    ("HM", "Heard Island and McDonald Islands"),
    ("HN", "Honduras"),
    ("HK", "Hong Kong"),
    ("HU", "Hungary"),
    ("IS", "Iceland"),
    ("IN", "India"),
    ("ID", "Indonesia"),
    ("IR", "Iran"),
    ("IQ", "Iraq"),
    ("IE", "Ireland"),
    ("IM", "Isle of Man"),
    ("IL", "Israel"),
    ("IT", "Italy"),
    ("JM", "Jamaica"),
    ("JP", "Japan"),
    ("JE", "Jersey"),
    ("JO", "Jordan"),
    ("KZ", "Kazakhstan"),
    ("KE", "Kenya"),
    ("KI", "Kiribati"),
    ("XK", "Kosovo"),
    ("KW", "Kuwait"),
    ("KG", "Kyrgyzstan"),
    ("LA", "Laos"),
    ("LV", "Latvia"),
    ("LB", "Lebanon"),
    ("LS", "Lesotho"),
    ("LR", "Liberia"),
    ("LY", "Libya"),
    ("LI", "Liechtenstein"),
    ("LT", "Lithuania"),
    ("LU", "Luxembourg"),
    ("MO", "Macau"),
    ("MK", "North Macedonia"),
    ("MG", "Madagascar"),
    ("MW", "Malawi"),
    ("MY", "Malaysia"),
    ("MV", "Maldives"),
    ("ML", "Mali"),
    ("MT", "Malta"),
    ("MH", "Marshall Islands"),
    ("MR", "Mauritania"),
    ("MU", "Mauritius"),
    ("YT", "Mayotte"),
    ("MX", "Mexico"),
    ("FM", "Micronesia"),
    ("MD", "Moldova"),
    ("MC", "Monaco"),
    ("MN", "Mongolia"),
    ("ME", "Montenegro"),
    ("MS", "Montserrat"),
    ("MA", "Morocco"),
    ("MZ", "Mozambique"),
    ("MM", "Myanmar"),
    ("NA", "Namibia"),
    ("NR", "Nauru"),
    ("NP", "Nepal"),
    ("NL", "Netherlands"),
    ("NC", "New Caledonia"),
    ("NZ", "New Zealand"),
    ("NI", "Nicaragua"),
    ("NE", "Niger"),
    ("NG", "Nigeria"),
    ("NU", "Niue"),
    ("NF", "Norfolk Island"),
    ("KP", "North Korea"),
    ("MP", "Northern Mariana Islands"),
    ("NO", "Norway"),
    ("OM", "Oman"),
    ("PK", "Pakistan"),
    ("PW", "Palau"),
    ("PS", "Palestine"),
    ("PA", "Panama"),
    ("PG", "Papua New Guinea"),
    ("PY", "Paraguay"),
    ("PE", "Peru"),
    ("PH", "Philippines"),
    ("PN", "Pitcairn Islands"),
    ("PL", "Poland"),
    ("PT", "Portugal"),
    ("PR", "Puerto Rico"),
    ("QA", "Qatar"),
    ("CG", "Republic of the Congo"),
    ("RO", "Romania"),
    ("RU", "Russia"),
    ("RW", "Rwanda"),
    ("RE", "Réunion"),
    ("BL", "Saint Barthélemy"),
    ("SH", "Saint Helena"),
    ("KN", "Saint Kitts and Nevis"),
    ("LC", "Saint Lucia"),
    ("MF", "Saint Martin"),
    ("PM", "Saint Pierre and Miquelon"),
    ("VC", "Saint Vincent and the Grenadines"),
    ("WS", "Samoa"),
    ("SM", "San Marino"),
    ("ST", "Sao Tome and Principe"),
    ("SA", "Saudi Arabia"),
    ("SN", "Senegal"),
    ("RS", "Serbia"),
    ("SC", "Seychelles"),
    ("SL", "Sierra Leone"),
    ("SG", "Singapore"),
    ("SX", "Sint Maarten"),
    ("SK", "Slovakia"),
    ("SI", "Slovenia"),
    ("SB", "Solomon Islands"),
    ("SO", "Somalia"),
    ("ZA", "South Africa"),
    ("GS", "South Georgia and the South Sandwich Islands"),
    ("KR", "South Korea"),
    ("SS", "South Sudan"),
    ("ES", "Spain"),
    ("LK", "Sri Lanka"),
    ("SD", "Sudan"),
    ("SR", "Suriname"),
    ("SJ", "Svalbard and Jan Mayen"),
    ("SE", "Sweden"),
    ("CH", "Switzerland"),
    ("SY", "Syria"),
    ("TW", "Taiwan"),
    ("TJ", "Tajikistan"),
    ("TZ", "Tanzania"),
    ("TH", "Thailand"),
    ("TL", "Timor-Leste"),
    ("TG", "Togo"),
    ("TK", "Tokelau"),
    ("TO", "Tonga"),
    ("TT", "Trinidad and Tobago"),
    ("TN", "Tunisia"),
    ("TR", "Turkey"),
    ("TM", "Turkmenistan"),
    ("TC", "Turks and Caicos Islands"),
    ("TV", "Tuvalu"),
    ("UG", "Uganda"),
    ("UA", "Ukraine"),
    ("AE", "United Arab Emirates"),
    ("GB", "United Kingdom"),
    ("US", "United States"),
    ("UM", "United States Minor Outlying Islands"),
    ("VI", "United States Virgin Islands"),
    ("UY", "Uruguay"),
    ("UZ", "Uzbekistan"),
    ("VU", "Vanuatu"),
    ("VA", "Vatican City"),
    ("VE", "Venezuela"),
    ("VN", "Vietnam"),
    ("WF", "Wallis and Futuna"),
    ("EH", "Western Sahara"),
    ("YE", "Yemen"),
    ("ZM", "Zambia"),
    ("ZW", "Zimbabwe"),
    ("AX", "Åland Islands"),
]
COUNTRY_MAP = {c: n for c, n in COUNTRY_CHOICES}

# =========================
# MAPPINGS (codes -> text)
# =========================
GENDER_MAP = {0: "female", 1: "male", 2: "diverse"}

ETHNICITY_MAP = {
    0: "White",
    1: "Asian",
    2: "Black / African descent",
    3: "Hispanic or Latino",
    4: "Multiple ethnic groups",
    5: "Middle Eastern / North African",
    6: "Other ethnic group",
}

STUDENT_STATUS_MAP = {
    1: "Full-time student",
    2: "Part-time student",
    3: "Not currently a student",
}

EMPLOYMENT_STATUS_MAP = {
    0: "Employed full-time",
    1: "Employed part-time",
    2: "Self-employed",
    3: "Unemployed",
    4: "Not in the labor force (e.g. retired, caregiver, unable to work)",
}

QUALIFICATION_MAP = {
    1: "No formal education",
    2: "Secondary school (GCSEs/ O Levels)",
    3: "Further education (A Levels, BTEC, apprenticeship)",
    4: "Bachelor's degree",
    5: "Master's degree",
    6: "PhD or higher",
}


# =========================
# Helpers
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_str(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s != "" else None

def safe_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None

def safe_folder_name(s: str) -> str:
    bad = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip()

def decode_with_map(val, mapping: Dict[int, str]) -> Optional[str]:
    iv = safe_int(val)
    if iv is None:
        return None
    return mapping.get(iv, str(iv))

def country_code_to_name(code) -> Optional[str]:
    code = safe_str(code)
    if not code:
        return None
    return COUNTRY_MAP.get(code.upper(), code)

def normalize_message_bank(obj: Any) -> Dict[str, Dict[str, str]]:
    """
    JSON is stored like: [ { "1": {...}, "2": {...}, "3": {...} } ]
    Returns dict with keys "1","2","3".
    """
    if isinstance(obj, list) and len(obj) > 0:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("Message bank JSON must be a dict or [dict].")
    return obj

def extract_ordered_messages(
    bank: Dict[str, Dict[str, str]],
    package_name: str,
    order_index_0to5: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Prolific message_order is 0..5, mapping directly into PERMS.
    """
    package_name = safe_str(package_name) or ""

    idx = 0 if order_index_0to5 is None else int(order_index_0to5)
    idx = max(0, min(5, idx))  # clamp
    perm = PERMS[idx]          # direct

    raw = {}
    for sid in ("1", "2", "3"):
        entry = bank.get(sid, {})
        raw[int(sid)] = safe_str(entry.get(package_name))

    ordered_msgs = []
    for shown_pos, source_id in enumerate(perm, start=1):
        ordered_msgs.append({
            "shown_comment_id": shown_pos,
            "source_message_id": source_id,
            "package": package_name,
            "text": raw.get(source_id),
        })
    return ordered_msgs


# =========================
# Main exporter
# =========================
def export_prolific_personas(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    ubi_bank = normalize_message_bank(load_json(UBI_MESSAGES_PATH))
    penalty_bank = normalize_message_bank(load_json(PENALTY_MESSAGES_PATH))
    wl_bank = normalize_message_bank(load_json(WEIGHT_LOSS_MESSAGES_PATH))

    MESSAGE_BANKS = {
        "UBI": ubi_bank,
        "penalty": penalty_bank,
        "weight_loss": wl_bank,
    }

    PID_COL = "participant.label"
    big_cols = [f"survey.1.player.big{i}" for i in range(1, 11)]

    for _, row in df.iterrows():
        pid = safe_str(row.get(PID_COL))
        if not pid:
            continue

        persona_dir = os.path.join(out_dir, safe_folder_name(pid))
        os.makedirs(persona_dir, exist_ok=True)

        # -------------------------
        # DEMOGRAPHIC JSON
        # -------------------------
        # Your description says survey.1.player.age holds gender codes, which is inconsistent.
        # We keep raw age + attempt decoding from survey.1.player.gender if that column exists.
        age_raw = row.get("survey.1.player.age")

        demographic = {
            "participant_id": pid,

            "age_raw": age_raw,
            "age": safe_int(age_raw),

            "gender_raw": row.get("survey.1.player.gender"),
            "gender": decode_with_map(row.get("survey.1.player.gender"), GENDER_MAP),

            "ethnicity_raw": row.get("survey.1.player.ethnicity"),
            "ethnicity": decode_with_map(row.get("survey.1.player.ethnicity"), ETHNICITY_MAP),

            "country_of_residency_code": safe_str(row.get("survey.1.player.country_of_residency")),
            "country_of_residency": country_code_to_name(row.get("survey.1.player.country_of_residency")),

            "country_of_origin_code": safe_str(row.get("survey.1.player.country_of_origin")),
            "country_of_origin": country_code_to_name(row.get("survey.1.player.country_of_origin")),

            "student_status_raw": row.get("survey.1.player.student_status"),
            "student_status": decode_with_map(row.get("survey.1.player.student_status"), STUDENT_STATUS_MAP),

            "employment_status_raw": row.get("survey.1.player.employment_status"),
            "employment_status": decode_with_map(row.get("survey.1.player.employment_status"), EMPLOYMENT_STATUS_MAP),

            "occupation_field": safe_str(row.get("survey.1.player.occupation_field")),

            "highest_qualification_raw": row.get("survey.1.player.highest_qualification"),
            "highest_qualification": decode_with_map(row.get("survey.1.player.highest_qualification"), QUALIFICATION_MAP),

            "personality": {c: row.get(c) for c in big_cols if c in df.columns},
        }

        # -------------------------
        # STUDY DATA JSON (non-demographic + exact messages)
        # -------------------------
        topic_order = [
            safe_str(row.get("survey.1.player.first_topic")),
            safe_str(row.get("survey.1.player.second_topic")),
            safe_str(row.get("survey.1.player.third_topic")),
        ]
        topic_order = [t for t in topic_order if t]

        statements = {
            "UBI": safe_str(row.get("survey.1.player.UBI_statement_formulation")),
            "penalty": safe_str(row.get("survey.1.player.penalty_statement_formulation")),
            "weight_loss": safe_str(row.get("survey.1.player.weight_loss_statement_formulation")),
        }

        def topic_block(topic_name: str) -> Dict[str, Any]:
            pkg = safe_str(row.get(f"survey.1.player.package_{topic_name}"))
            order_idx = safe_int(row.get(f"survey.1.player.message_order_{topic_name}"))
            if order_idx is None:
                order_idx = 0  # prolific default if missing

            init_belief = row.get(f"survey.1.player.init_belief_{topic_name}")
            familiarity = row.get(f"survey.1.player.familiarity_{topic_name}")

            shown_messages = extract_ordered_messages(
                bank=MESSAGE_BANKS[topic_name],
                package_name=pkg or "",
                order_index_0to5=order_idx,
            )

            # clamp index for storing perm
            idx = max(0, min(5, int(order_idx)))

            return {
                "topic": topic_name,
                "statement_formulation": statements.get(topic_name),

                "init_belief": init_belief,
                "familiarity": familiarity,

                "package": pkg,
                "message_order_index": idx,          # 0..5 (Prolific)
                "message_order_perm": list(PERMS[idx]),

                "shown_messages": shown_messages,
            }

        study_data = {
            "participant_id": pid,
            "topic_order": topic_order,
            "topics": [
                topic_block("UBI"),
                topic_block("penalty"),
                topic_block("weight_loss"),
            ],
            "raw_fields_used": {
                "survey.1.player.first_topic": safe_str(row.get("survey.1.player.first_topic")),
                "survey.1.player.second_topic": safe_str(row.get("survey.1.player.second_topic")),
                "survey.1.player.third_topic": safe_str(row.get("survey.1.player.third_topic")),

                "survey.1.player.package_UBI": safe_str(row.get("survey.1.player.package_UBI")),
                "survey.1.player.message_order_UBI": safe_int(row.get("survey.1.player.message_order_UBI")),

                "survey.1.player.package_penalty": safe_str(row.get("survey.1.player.package_penalty")),
                "survey.1.player.message_order_penalty": safe_int(row.get("survey.1.player.message_order_penalty")),

                "survey.1.player.package_weight_loss": safe_str(row.get("survey.1.player.package_weight_loss")),
                "survey.1.player.message_order_weight_loss": safe_int(row.get("survey.1.player.message_order_weight_loss")),
            }
        }

        with open(os.path.join(persona_dir, "demographic.json"), "w", encoding="utf-8") as f:
            json.dump(demographic, f, ensure_ascii=False, indent=2)

        with open(os.path.join(persona_dir, "study_data.json"), "w", encoding="utf-8") as f:
            json.dump(study_data, f, ensure_ascii=False, indent=2)

        print(f"Exported persona {pid} -> {persona_dir}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, encoding="utf-8", low_memory=False)
    export_prolific_personas(df, BASE_OUTPUT_DIR)
    print("Done.")
