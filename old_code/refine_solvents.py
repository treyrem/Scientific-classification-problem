import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import openai

# Configure logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"refine_solvents_{timestamp}.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()

# Load your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

# Files
INPUT_CSV = "products_with_updated_solvents_categories.csv"
OUTPUT_CSV = "products_refined_solvents.csv"

# Prompt template for subclassifying solvents & solutions
SOLVENT_PROMPT = (
    "You are given a chemical name. Classify it into exactly one of the following sub-subcategories:\n"
    "1) Organic Solvent\n"
    "2) Aqueous Solution\n"
    "3) Dilution Buffer\n"
    'Respond ONLY with JSON like {"sub_subcategory":"<one of the three>"}.\n'
    "Chemical: {name}\n"
)

# Retry configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 2


def classify_solvent(name: str) -> str:
    """
    Calls OpenAI to classify a solvent into a sub-subcategory.
    Returns one of 'Organic Solvent', 'Aqueous Solution', 'Dilution Buffer', or 'Unknown'.
    """
    prompt = SOLVENT_PROMPT.format(name=name)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful lab reagent assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=50,
            )
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
            sub = data.get("sub_subcategory")
            if sub not in {"Organic Solvent", "Aqueous Solution", "Dilution Buffer"}:
                raise ValueError(f"Unexpected category: {sub}")
            return sub
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for '{name}': {e}")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_FACTOR ** (attempt - 1))
            else:
                logger.error(f"All retries failed for '{name}'. Marking as Unknown.")
                return "Unknown"


def main():
    # Load data
    df = pd.read_csv(INPUT_CSV)
    # Filter to solvents
    mask = df["subcategory"] == "Solvents & Solutions"
    solvents = df.loc[mask, :].copy()
    logger.info(f"Found {len(solvents)} rows in 'Solvents & Solutions'.")

    # Classify in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_name = {
            executor.submit(classify_solvent, name): name
            for name in solvents["product_name"]
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                sub_sub = future.result()
            except Exception as exc:
                logger.error(f"Error classifying {name}: {exc}")
                sub_sub = "Unknown"
            results[name] = sub_sub

    # Assign results
    solvents["sub_subcategory"] = solvents["product_name"].map(results)

    # Merge back
    df.loc[mask, "sub_subcategory"] = solvents["sub_subcategory"].values

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Refined solvents saved to {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
