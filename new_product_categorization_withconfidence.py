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

# ─── CONFIG
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_with_categories_n_confidence.csv"
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

# MANUAL OVERRIDES
# Keeping only essential overrides for products that are commonly misclassified
OVERRIDES: List[Tuple[Pattern, Tuple[str, str, str]]] = [
    (
        re.compile(r"lipofectamine", re.I),
        (
            "Cell Culture & Biologicals",
            "Cell Culture Media & Reagents",
            "Transfection Reagents",
        ),
    ),
]

# ─── UTILITIES


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─── CLASSIFICATION
def classify_product(name: str) -> Dict[str, str]:
    for pattern, (top, sub, sub_sub) in OVERRIDES:
        if pattern.search(name):
            return {
                "category": top,
                "subcategory": sub,
                "sub_subcategory": sub_sub,
                "confidence": "High",
            }

    # 2) LLM classification
    prompt = (
        "You are a life-science cataloguing assistant.\n"
        "Given a product name, assign ONE category, ONE subcategory, and ONE sub-subcategory from the structure below.\n"
        "Additionally, provide your confidence in this classification as High, Medium, or Low.\n"
        "If you cannot determine an appropriate sub-subcategory, leave that field empty but still provide category and subcategory.\n"
        "Respond ONLY with raw JSON.\n\n"
        "Examples:\n"
        'Product name: "Anti-human CD3 monoclonal antibody"\n'
        'Response: {"category":"Antibodies", "subcategory":"Primary Antibodies", "sub_subcategory":"Monoclonal Antibodies", "confidence":"High"}\n\n'
        'Product name: "SYBR Green qPCR Master Mix"\n'
        'Response: {"category":"Nucleic Acid Products", "subcategory":"Oligos, Primers & Genes", "sub_subcategory":"Probes", "confidence":"Medium"}\n\n'
        'Product name: "General Lab Equipment Item"\n'
        'Response: {"category":"Lab Equipment", "subcategory":"General Lab Equipment", "sub_subcategory":"", "confidence":"Low"}\n\n'
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

        j = json.loads(stripped)

        top = j.get("category", "Other")
        sub = j.get("subcategory", "")
        sub_sub = j.get("sub_subcategory", "")
        confidence = j.get("confidence", "Low")  # Default to Low if not provided

        if top not in TOP_CATEGORIES:
            confidence = "Low"
            top, sub, sub_sub = "Other", "", ""

        if top == "Other":
            confidence = "Low"

        if sub not in CATEGORY_STRUCTURE.get(top, {}):
            confidence = "Low"
            available_subs = list(CATEGORY_STRUCTURE.get(top, {}).keys())
            sub = available_subs[0] if available_subs else ""

        if sub_sub and sub_sub not in CATEGORY_STRUCTURE.get(top, {}).get(sub, []):
            sub_sub = ""
            if confidence == "High":
                confidence = (
                    "Medium"  # Downgrade confidence if sub-subcategory incorrect
                )

        return {
            "category": top,
            "subcategory": sub,
            "sub_subcategory": sub_sub,
            "confidence": confidence,
        }

    except json.JSONDecodeError:
        logger.warning(f"JSON parse error for '{name}': {text!r}")
        return {
            "category": "Other",
            "subcategory": "",
            "sub_subcategory": "",
            "confidence": "Low",
        }
    except Exception as e:
        logger.error(f"Error classifying '{name}': {e}")
        return {
            "category": "Other",
            "subcategory": "",
            "sub_subcategory": "",
            "confidence": "Low",
        }


# MAIN
def main():
    CHECKPOINT_FILE = (
        r"C:\LabGit\150citations classification\checkpoints\checkpoint_34500.csv"
    )

    if os.path.exists(CHECKPOINT_DIR) and not os.path.isdir(CHECKPOINT_DIR):
        logger.error(
            f"'{CHECKPOINT_DIR}' exists but is not a directory. Please remove or rename this file."
        )
        exit(1)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        # Find the first row with empty 'category' to resume from there
        try:
            start_idx = df[df["category"].isna() | (df["category"] == "")].index[0]
        except IndexError:
            logger.info("Checkpoint file is fully processed. Exiting.")
            return
        logger.info(f"Resuming from checkpoint at index {start_idx}")
    else:
        df = pd.read_csv(INPUT_CSV)
        df["category"] = ""
        df["subcategory"] = ""
        df["sub_subcategory"] = ""
        df["confidence"] = ""
        start_idx = 0
        logger.info(f"No checkpoint found, starting classification from beginning.")

    total = len(df)
    for idx in tqdm(range(start_idx, total), desc="Classifying"):
        name = df.at[idx, "Name"]
        result = classify_product(name)
        df.at[idx, "category"] = result["category"]
        df.at[idx, "subcategory"] = result["subcategory"]
        df.at[idx, "sub_subcategory"] = result["sub_subcategory"]
        df.at[idx, "confidence"] = result["confidence"]

        if (idx + 1) % CHECKPOINT_FREQ == 0 or idx == total - 1:
            cp = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx+1}.csv")
            df.to_csv(cp, index=False)
            logger.info(f"Saved checkpoint: {cp}")

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Classification complete. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
