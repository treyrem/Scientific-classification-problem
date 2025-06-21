import pandas as pd
import json
import logging
import os
import re
from typing import Dict, List, Tuple, Pattern
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# ─── CONFIG
INPUT_CSV = (
    "C:/LabGit/150citations classification/products_with_categories_with_confidence.csv"
)
OUTPUT_CSV = "prod_w_cat_n_conf.csv"

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

# ─── EXPANDED CATEGORY STRUCTURE ─────────────────────────────────────────────
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
            "Microplate Readers",
            "Cell Sorters",
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
        "Cell Culture Supplements": ["Media Supplements", "Growth Factors"],
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
            "Serum",
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
            "Hitrap Columns",
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

# ─── BRAND/MODEL SYNONYMS MAPPING ────────────────────────────────────────────
BRAND_SYNONYMS: Dict[Pattern, Tuple[str, str, str]] = {
    re.compile(r"hitrap", re.I): (
        "Chromatography & Separation",
        "Chromatography Systems & Columns",
        "Hitrap Columns",
    ),
    re.compile(r"synergy", re.I): (
        "Lab Equipment",
        "Analytical Instruments",
        "Microplate Readers",
    ),
    re.compile(r"facsaria", re.I): (
        "Lab Equipment",
        "Analytical Instruments",
        "Cell Sorters",
    ),
    re.compile(r"ckx41", re.I): (
        "Lab Equipment",
        "Analytical Instruments",
        "Microscopes",
    ),
    re.compile(r"golgiplug", re.I): (
        "Chemicals & Reagents",
        "Biochemical Reagents",
        "Buffers",
    ),
    re.compile(r"beadchip", re.I): (
        "Assay Kits",
        "ELISA & Immunoassay Kits",
        "Multiplex Assays",
    ),
    re.compile(r"jsm\d+", re.I): (
        "Lab Equipment",
        "Analytical Instruments",
        "Imaging Systems",
    ),
    re.compile(r"zen blue", re.I): (
        "Lab Equipment",
        "Analytical Instruments",
        "Imaging Systems",
    ),
    re.compile(r"vt1000", re.I): (
        "Lab Equipment",
        "Analytical Instruments",
        "Sequencers",
    ),
    re.compile(r"d12492", re.I): (
        "Chemicals & Reagents",
        "Small Molecules & Inhibitors",
        "Chemical Compounds",
    ),
    re.compile(r"alpha mem", re.I): (
        "Cell Culture & Biologicals",
        "Cell Culture Media & Reagents",
        "Growth Media",
    ),
    re.compile(r"du145", re.I): (
        "Cell Culture & Biologicals",
        "Cells, Tissues & Strains",
        "Cell Lines",
    ),
}

OVERRIDES: List[Tuple[Pattern, Tuple[str, str, str]]] = list(BRAND_SYNONYMS.items())

# ─── UTILITIES ─────────────────────────────────────────────────────────────


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─── CLASSIFICATION ─────────────────────────────────────────────────────────
def classify_product(name: str) -> Dict[str, str]:
    for pattern, (top, sub, sub_sub) in OVERRIDES:
        if pattern.search(name):
            return {
                "category": top,
                "subcategory": sub,
                "sub_subcategory": sub_sub,
                "confidence": "High",
            }

    prompt = (
        "You are a life-science cataloguing assistant.\n"
        "Given a product name, assign ONE category, ONE subcategory, and ONE sub-subcategory from the structure below.\n"
        "Additionally, provide your confidence in this classification as High, Medium, or Low.\n"
        "For known brand/model names, prioritize mapping based on provided synonym table.\n"
        "Respond ONLY with raw JSON.\n\n"
        f"Valid CATEGORIES: {', '.join(CATEGORY_STRUCTURE.keys())}\n"
        f'Product name: "{name}"\n'
        "Response:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        data = json.loads(strip_code_fences(resp.choices[0].message.content))
    except Exception:
        return {
            "category": "Other",
            "subcategory": "",
            "sub_subcategory": "",
            "confidence": "Low",
        }

    # Ensure all keys exist
    data.setdefault("category", "Other")
    data.setdefault("subcategory", "")
    data.setdefault("sub_subcategory", "")
    data.setdefault("confidence", "Low")

    # Validate category
    if data["category"] not in CATEGORY_STRUCTURE:
        return {
            "category": "Other",
            "subcategory": "",
            "sub_subcategory": "",
            "confidence": "Low",
        }
    return data


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV)
    mask = df["category"] == "Other"
    other_indices = df[mask].index.tolist()

    for idx in tqdm(other_indices, desc="Reclassifying 'Other' items"):
        name = df.at[idx, "Name"]
        result = classify_product(name)
        df.at[idx, "category"] = result["category"]
        df.at[idx, "subcategory"] = result["subcategory"]
        df.at[idx, "sub_subcategory"] = result["sub_subcategory"]
        df.at[idx, "confidence"] = result["confidence"]

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Re-classification complete. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
