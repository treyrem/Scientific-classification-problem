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
OUTPUT_CSV = "products_with_categories_Claude.csv"
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

CATEGORY_STRUCTURE = {
    "Antibodies": {
        "Primary Antibodies": ["Monoclonal Antibodies", "Polyclonal Antibodies"],
        "Secondary Antibodies": ["Enzyme Conjugates", "Fluorophore Conjugates"],
    },
    "Assay Kits": {
        "ELISA & Immunoassay Kits": ["ELISA Kits", "Multiplex Assays"],
        "Cell-Based & Functional Assay Kits": [
            "Cell Viability Kits",
            "Apoptosis Kits",
            "Metabolism Kits",
            "Enzyme Activity Kits",
        ],
    },
    "Lab Equipment": {
        "Analytical Instruments": [
            "Microscopes",
            "Imaging Systems",
            "Sequencers",
            "Spectrometers",
            "Flow Cytometers",
        ],
        "General Lab Equipment": [
            "Centrifuges",
            "Incubators",
            "Shakers",
            "Thermal Cyclers",
            "PCR Machines",
        ],
    },
    "Lab Supplies & Consumables": {
        "Labware (Plastic & Glass)": [
            "Tubes",
            "Pipette Tips",
            "Plates",
            "Flasks",
            "Bottles",
        ],
        "Filtration & Pipetting Supplies": [
            "Filters",
            "Syringes",
            "Pipettes",
            "Pipettors",
        ],
    },
    "Chemicals & Reagents": {
        "Biochemical Reagents": ["Enzymes", "Substrates", "Dyes & Stains", "Buffers"],
        "Small Molecules & Inhibitors": [
            "Chemical Compounds",
            "Inhibitors",
            "Antibiotics",
        ],
        "Solvents & Solutions": [
            "Organic Solvents",
            "Aqueous Solutions",
            "Dilution Buffers",
        ],
    },
    "Proteins & Peptides": {
        "Recombinant Proteins & Enzymes": [
            "Cytokines",
            "Growth Factors",
            "Expressed Proteins",
        ],
        "Peptides & Protein Fragments": [
            "Research-Grade Peptides",
            "Protein Fragments",
            "Biomolecular Standards",
        ],
    },
    "Nucleic Acid Products": {
        "Oligos, Primers & Genes": [
            "DNA Oligonucleotides",
            "RNA Oligonucleotides",
            "Primers",
            "Probes",
            "Gene Synthesis Products",
        ],
        "RNAi & Gene Editing Tools": ["siRNA", "shRNA", "CRISPR/Cas Reagents"],
    },
    "Cell Culture & Biologicals": {
        "Cell Culture Media & Reagents": [
            "Growth Media",
            "Sera",
            "Supplements",
            "Transfection Reagents",
        ],
        "Cells, Tissues & Strains": [
            "Cell Lines",
            "Primary Cells",
            "Microorganisms",
            "Tissue Products",
            "Blood Products",
        ],
    },
    "Chromatography & Separation": {
        "Chromatography Systems & Columns": [
            "HPLC Systems",
            "FPLC Systems",
            "GC Units",
            "Columns",
            "Cartridges",
        ],
        "Electrophoresis & Western Blot Supplies": [
            "Gel Electrophoresis Apparatus",
            "Gels",
            "Membranes",
        ],
    },
    "Software & Informatics": {
        "Data Analysis Software": [
            "Bioinformatics Tools",
            "Imaging Analysis Software",
            "Instrument Analysis Software",
        ],
        "Laboratory Management Software": ["LIMS", "Workflow Management Software"],
    },
    "Services": {
        "Custom Research/Production Services": [
            "Antibody Production",
            "Peptide Synthesis",
            "DNA Sequencing",
        ],
        "Analytical & CRO Services": [
            "Contract Research",
            "Sample Testing",
            "Sequencing",
            "Analysis Services",
        ],
    },
    "Animal Models": {
        "Mice Models": ["Transgenic Mice", "Knockout Mice", "Humanized Mice"],
        "Other Animal Models": ["Rat Models", "Zebrafish Models", "Drosophila Models"],
    },
    "Other": {},
}

# Extract top-level categories
TOP_CATEGORIES = list(CATEGORY_STRUCTURE.keys())

# Pre-format lists for prompt
NUMBERED_CATEGORIES = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(TOP_CATEGORIES))

# Format subcategories and sub-subcategories for prompt
FORMATTED_STRUCTURE = []
for top_cat, subcats in CATEGORY_STRUCTURE.items():
    if subcats:  # Skip empty categories like "Other"
        for subcat, sub_subcats in subcats.items():
            sub_subcat_str = (
                ", ".join(sub_subcats) if sub_subcats else "No sub-subcategories"
            )
            FORMATTED_STRUCTURE.append(f"- {top_cat} > {subcat}: {sub_subcat_str}")

FORMATTED_SUBCATS = "\n".join(FORMATTED_STRUCTURE)

# ─── MANUAL OVERRIDES ──────────────────────────────────────────────────────────
# Keeping only essential overrides for products that are commonly misclassified
OVERRIDES: List[Tuple[Pattern, Tuple[str, str, str]]] = [
    # Essential overrides for brand-specific products that might be unclear to LLM
    (
        re.compile(r"lipofectamine", re.I),
        (
            "Cell Culture & Biologicals",
            "Cell Culture Media & Reagents",
            "Transfection Reagents",
        ),
    ),
]

# ─── UTILITIES ──────────────────────────────────────────────────────────────────


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─── CLASSIFICATION ─────────────────────────────────────────────────────────────
def classify_product(name: str) -> Dict[str, str]:
    # 1) Manual overrides (minimal, only for essential cases)
    for pattern, (top, sub, sub_sub) in OVERRIDES:
        if pattern.search(name):
            return {"category": top, "subcategory": sub, "sub_subcategory": sub_sub}

    # 2) LLM classification
    prompt = (
        "You are a life-science cataloguing assistant.\n"
        "Given a product name, assign ONE category, ONE subcategory, and ONE sub-subcategory from the structure below.\n"
        "If you cannot determine an appropriate sub-subcategory, leave that field empty but still provide category and subcategory.\n"
        "Respond ONLY with raw JSON.\n\n"
        "Examples:\n"
        'Product name: "Anti-human CD3 monoclonal antibody"\n'
        'Response: {"category":"Antibodies", "subcategory":"Primary Antibodies", "sub_subcategory":"Monoclonal Antibodies"}\n\n'
        'Product name: "SYBR Green qPCR Master Mix"\n'
        'Response: {"category":"Nucleic Acid Products", "subcategory":"Oligos, Primers & Genes", "sub_subcategory":"Probes"}\n\n'
        'Product name: "General Lab Equipment Item"\n'
        'Response: {"category":"Lab Equipment", "subcategory":"General Lab Equipment", "sub_subcategory":""}\n\n'
        f"Valid CATEGORIES:\n{NUMBERED_CATEGORIES}\n\n"
        f"Valid SUBCATEGORIES and SUB-SUBCATEGORIES:\n{FORMATTED_SUBCATS}\n\n"
        f'Product name: "{name}"\n'
        "Response:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        stripped = strip_code_fences(text)

        # 3) JSON parse
        j = json.loads(stripped)

        top = j.get("category", "Other")
        sub = j.get("subcategory", "")
        sub_sub = j.get("sub_subcategory", "")

        # Validate classification
        if top not in TOP_CATEGORIES:
            logger.warning(f"Invalid top category '{top}' for '{name}'. Using 'Other'.")
            return {"category": "Other", "subcategory": "", "sub_subcategory": ""}

        if top == "Other":
            return {"category": "Other", "subcategory": "", "sub_subcategory": ""}

        # If subcategory is invalid, try to find the best match or use first available
        if sub not in CATEGORY_STRUCTURE.get(top, {}):
            available_subs = list(CATEGORY_STRUCTURE.get(top, {}).keys())
            if available_subs:
                sub = available_subs[0]  # Use first available subcategory
                logger.warning(
                    f"Invalid subcategory '{j.get('subcategory', '')}' for category '{top}' and product '{name}'. Using '{sub}'."
                )
            else:
                sub = ""

        # If sub-subcategory is provided but invalid, just leave it empty (don't force a value)
        if (
            sub
            and sub_sub
            and sub_sub not in CATEGORY_STRUCTURE.get(top, {}).get(sub, [])
        ):
            logger.info(
                f"Invalid sub-subcategory '{sub_sub}' for '{top}' > '{sub}' and product '{name}'. Leaving empty."
            )
            sub_sub = ""

        return {"category": top, "subcategory": sub, "sub_subcategory": sub_sub}

    except json.JSONDecodeError:
        logger.warning(f"JSON parse error for '{name}': {text!r}")
        return {"category": "Other", "subcategory": "", "sub_subcategory": ""}
    except Exception as e:
        logger.error(f"Error classifying '{name}': {e}")
        return {"category": "Other", "subcategory": "", "sub_subcategory": ""}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Ensure checkpoint dir is a directory
    if os.path.exists(CHECKPOINT_DIR) and not os.path.isdir(CHECKPOINT_DIR):
        logger.error(
            f"'{CHECKPOINT_DIR}' exists but is not a directory. Please remove or rename this file."
        )
        exit(1)
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
        df["sub_subcategory"] = ""
        start_idx = 0
        logger.info(f"Starting classification of {len(df)} products")

    # Loop and checkpoint
    total = len(df)
    for idx in tqdm(range(start_idx, total), desc="Classifying"):
        name = df.at[idx, "Name"]
        result = classify_product(name)
        df.at[idx, "category"] = result["category"]
        df.at[idx, "subcategory"] = result["subcategory"]
        df.at[idx, "sub_subcategory"] = result["sub_subcategory"]

        if (idx + 1) % CHECKPOINT_FREQ == 0 or idx == total - 1:
            cp = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx+1}.csv")
            df.to_csv(cp, index=False)
            logger.info(f"Saved checkpoint: {cp}")

    # Final output
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Classification complete. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
