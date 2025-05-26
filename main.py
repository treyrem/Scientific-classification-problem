import os
import glob
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging
from time import sleep
from openai import OpenAI

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# load OpenAI API key

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

# Input and output files
def find_latest_checkpoint():
    
    files = glob.glob("checkpoint_*.csv") + glob.glob("checkpoint_*.xlsx")
    if not files:
        return None
    
    idxs = []
    for f in files:
        stem = Path(f).stem  
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            idxs.append((int(parts[1]), f))
    if not idxs:
        return None
    # pick the file with the highest index
    max_idx, max_file = max(idxs, key=lambda x: x[0])
    return max_file, max_idx


INPUT_CSV = "products_over_150_protocols_list.csv"
OUTPUT_CSV = "products_with_categories.csv"

# Category taxonomy (35 categories)
CATEGORIES = [
    "Antibodies",
    "Assay Kits",
    "Bioimaging & Microscopy",
    "Blood & Tissue Products",
    "Cell Biology",
    "Cell Culture Media & Reagents",
    "Cloning & Expression",
    "Immunochemicals",
    "Chromatography",
    "Flow Cytometry",
    "Liquid Handling",
    "Microplate Readers",
    "Molecular Biology",
    "Molecular Diagnostics",
    "Nucleic Acid Electrophoresis",
    "Nucleic Acid Purification",
    "PCR",
    "Protein Biochemistry",
    "RNAi Technology",
    "Sequencing & Genomics Tools",
    "Enzymes",
    "Inhibitors",
    "Chemicals & Reagents",
    "Plasticware & Consumables",
    "Centrifugation Equipment",
    "Filtration Systems",
    "Lab Automation & High-Throughput",
    "Laboratory Equipment",
    "Laboratory Services",
    "Life Science Software",
    "Translational Research",
    "Microorganisms & Cells",
    "Recombinant Proteins",
    "Cell Isolation",
    "Storage & Organization",
]
# Pre-format numbered categories
NUMBERED_CATEGORIES = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(CATEGORIES))


# Classification function
def classify_product(client, name: str) -> str:
    prompt = f"""
You are a life-science cataloguing assistant.
Given a product name, choose exactly one category from the numbered list below.
Respond with exactly the category name (no numbers, no extra text) and nothing else.

Valid categories (pick exactly one):
{NUMBERED_CATEGORIES}

Product name: \"{name}\"
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        cat = resp.choices[0].message.content.strip()
        if cat not in CATEGORIES:
            logger.warning(f"Unrecognized '{cat}' for '{name}'")
            variants = {
                "Cell Culture": "Cell Culture Media & Reagents",
                "Transfection & Transduction": "Cloning & Expression",
            }
            return variants.get(cat, "Other")
        return cat
    except Exception as e:
        logger.error(f"Error classifying '{name}': {e}")
        return "Other"


# Main pipeline
def main():
    # determine start from latest checkpoint or begin fresh
    cp = find_latest_checkpoint()
    if cp:
        cp_file, start_idx = cp
        # load checkpoint file, handling CSV or XLSX
        if cp_file.lower().endswith(".xlsx"):
            df = pd.read_excel(cp_file)
        else:
            df = pd.read_csv(cp_file)
        logger.info(f"Resuming from {cp_file} at index {start_idx}")
        existing = df["category"].tolist()[:start_idx]
    else:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Starting fresh: loaded {len(df)} products")
        df["category"] = "Other"
        start_idx = 0
        existing = []

    total = len(df)
    categories = existing.copy()

    # process remaining products
    for i in range(start_idx, total):
        name = df.at[i, "Name"] if "Name" in df.columns else ""
        cat = classify_product(client, name)
        categories.append(cat)
        if i < start_idx + 10:
            logger.info(f"{i+1}. '{name}' -> {cat}")
        sleep(1)
        if (i + 1) % 50 == 0:
            # write checkpoint as CSV for consistency
            df_tmp = df.copy()
            df_tmp["category"] = categories + [None] * (total - len(categories))
            cp_name = f"checkpoint_{i+1}.csv"
            df_tmp.to_csv(cp_name, index=False)
            logger.info(f"Checkpoint saved: {cp_name}")

    # finalize and save
    df["category"] = categories
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Completed. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


