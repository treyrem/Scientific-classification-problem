import pandas as pd
import json
import logging
import os
import re
from typing import Dict, List, Tuple, Pattern
from openai import OpenAI  # pip install openai
from tqdm import tqdm
import glob
from pathlib import Path
from dotenv import load_dotenv

# ─── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_CSV = "products_over_150_protocols_list.csv"
OUTPUT_CSV = "products_with_categories_Open_AI.csv"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 100  # save every N items

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_openai_key(env_file=None):
    env_path = Path(
        env_file or r"C:\LabGit\150citations classification\api_keys\OPEN_AI_KEY.env"
    )
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
    logger.error(f"OpenAI key not found in {env_path}")
    raise FileNotFoundError("OpenAI API key not found")


# instantiate new-style client
client = OpenAI(api_key=get_openai_key())

TOP_CATEGORIES = [
    "Biological Reagents",
    "Assay & Detection Kits",
    "Instruments & Equipment",
    "Consumables & Plasticware",
    "Chemicals & Small-Molecule Reagents",
    "Cell & Microbial Biology",
    "Proteomics & Protein Analysis",
    "Sequencing & Genomics",
    "Software & Services",
    "Safety & Waste Management",
]

SUBCATEGORIES: Dict[str, List[str]] = {
    "Biological Reagents": [
        "Antibodies",
        "Enzymes",
        "Recombinant Proteins & Growth Factors",
        "Primers & Oligonucleotides",
    ],
    "Assay & Detection Kits": [
        "qPCR Kits",
        "ELISA Kits",
        "NGS & Library Prep Kits",
        "Diagnostic Kits",
        "Nucleic Acid Purification Kits",
    ],
    "Instruments & Equipment": [
        "Microscopes & Imaging Systems",
        "Microplate Readers",
        "HPLC Systems & Columns",
        "Electrophoresis & Blotting Equipment",
        "Liquid Handling & Automation",
        "Centrifuges & Incubators",
        "Spectrophotometers",
        "Particle Size Analyzers",
        "Flow Cytometers",
        "PCR Instruments",
    ],
    "Consumables & Plasticware": [
        "Pipette Tips & Tubes",
        "Plates & Dishes",
        "Filters & Membranes",
        "Glassware & Cultureware",
    ],
    "Chemicals & Small-Molecule Reagents": [
        "Buffers & Solutions",
        "Solvents & Acids/Bases",
        "Dyes & Stains",
        "Specialty Chemicals",
        "Extraction Reagents",
    ],
    "Cell & Microbial Biology": [
        "Cell Culture Media",
        "Cell Lines & Primary Cells",
        "Microbial Strains & Growth Reagents",
        "Transfection & Transduction Reagents",
    ],
    "Proteomics & Protein Analysis": [
        "Mass Spectrometry Reagents",
        "Protein Assay Kits",
        "Protein Ladders & Markers",
    ],
    "Sequencing & Genomics": [
        "Sequencing Platforms & Adapters",
        "Genotyping & SNP Assays",
        "Transcriptomics & RNA-Seq Reagents",
    ],
    "Software & Services": [
        "Life Science Software",
        "Laboratory Information Management",
        "Contract Research Services",
    ],
    "Safety & Waste Management": [
        "Personal Protective Equipment",
        "Biohazard & Waste Disposal",
        "Lab Safety Supplies",
    ],
}

# Pre-format lists for prompt
NUMBERED_CATEGORIES = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(TOP_CATEGORIES))
NUMBERED_SUBCATS = "\n".join(
    f"- {top}: {', '.join(subs)}" for top, subs in SUBCATEGORIES.items()
)

# ─── MANUAL OVERRIDES ──────────────────────────────────────────────────────────
OVERRIDES: List[Tuple[Pattern, Tuple[str, str]]] = [
    # Manual overrides for specific products
    (
        re.compile(r"hitrap", re.I),
        ("Instruments & Equipment", "HPLC Systems & Columns"),
    ),
    (
        re.compile(r"glutamax", re.I),
        ("Cell & Microbial Biology", "Cell Culture Media"),
    ),
    (
        re.compile(r"alexa fluor", re.I),
        ("Chemicals & Small-Molecule Reagents", "Dyes & Stains"),
    ),
    (
        re.compile(r"cdna synthesis kit", re.I),
        ("Sequencing & Genomics", "Transcriptomics & RNA-Seq Reagents"),
    ),
    # Existing overrides
    (
        re.compile(r"(fbs|fetal bovine serum|fetal calf serum|serum)", re.I),
        ("Cell & Microbial Biology", "Cell Culture Media"),
    ),
    (
        re.compile(r"lipofectamine", re.I),
        ("Cell & Microbial Biology", "Transfection & Transduction Reagents"),
    ),
    (
        re.compile(r"triton x[- ]?100", re.I),
        ("Chemicals & Small-Molecule Reagents", "Specialty Chemicals"),
    ),
    (
        re.compile(r"protein assay kit|bca protein|bradford assay", re.I),
        ("Proteomics & Protein Analysis", "Protein Assay Kits"),
    ),
    (
        re.compile(r"flow cyt(om)?et(er)?", re.I),
        ("Instruments & Equipment", "Flow Cytometers"),
    ),
    (
        re.compile(r"hiseq|novaseq|miseq|nextseq|lightcycler|cycler|sequencing", re.I),
        ("Sequencing & Genomics", "Sequencing Platforms & Adapters"),
    ),
    (
        re.compile(r"nanodrop|spectrophotometer", re.I),
        ("Instruments & Equipment", "Spectrophotometers"),
    ),
    (
        re.compile(r"zeta?sizer|particle size", re.I),
        ("Instruments & Equipment", "Particle Size Analyzers"),
    ),
]

# ─── UTILITIES ──────────────────────────────────────────────────────────────────


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─── CLASSIFICATION ─────────────────────────────────────────────────────────────
def classify_product(name: str) -> Dict[str, str]:
    # 1) Manual overrides
    for pattern, (top, sub) in OVERRIDES:
        if pattern.search(name):
            return {"category": top, "subcategory": sub}

    # 2) LLM classification
    prompt = (
        "You are a life-science cataloguing assistant.\n"
        "Given a product name, assign ONE TOP-LEVEL category and ONE SUBCATEGORY below.\n"
        'Respond ONLY with raw JSON, e.g.: {{"category":"Assay & Detection Kits",'
        '"subcategory":"ELISA Kits"}}\n'
        f"Valid TOP-LEVEL categories:\n{NUMBERED_CATEGORIES}\n"
        f"Valid SUBCATEGORIES:\n{NUMBERED_SUBCATS}\n"
        f'Product name: "{name}"\n'
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    stripped = strip_code_fences(text)
    # 3) JSON parse
    try:
        j = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse error for '{name}': {text!r}")
        return {"category": "Other", "subcategory": ""}

    top = j.get("category", "Other")
    sub = j.get("subcategory", "")
    if top not in TOP_CATEGORIES or sub not in SUBCATEGORIES.get(top, []):
        logger.warning(f"Invalid classification {j!r} for '{name}'")
        return {"category": "Other", "subcategory": ""}
    return {"category": top, "subcategory": sub}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Ensure checkpoint dir is a directory
    if os.path.exists(CHECKPOINT_DIR) and not os.path.isdir(CHECKPOINT_DIR):
        logger.error(
            f"'{CHECKPOINT_DIR}' exists but is not a directory. Please remove or rename this file."
        )
        sys.exit(1)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load or start fresh
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        start_idx = len(df)
        logger.info(f"Resuming from index {start_idx}")
    else:
        df = pd.read_csv(INPUT_CSV)
        df["category"] = ""
        df["subcategory"] = ""
        start_idx = 0
        logger.info(f"Starting classification of {len(df)} products")

    # Loop and checkpoint
    total = len(df)
    for idx in tqdm(range(start_idx, total), desc="Classifying"):
        name = df.at[idx, "Name"]
        result = classify_product(name)
        df.at[idx, "category"] = result["category"]
        df.at[idx, "subcategory"] = result["subcategory"]

        if (idx + 1) % CHECKPOINT_FREQ == 0 or idx == total - 1:
            cp = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx+1}.csv")
            df.to_csv(cp, index=False)
            logger.info(f"Saved checkpoint: {cp}")

    # Final output
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Classification complete. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
