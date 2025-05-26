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


csv_path = (
    Path("C:/LabGit/150citations classification")
    / "csv results"
    / "products_with_categories_Claude - Copy - Copy.csv"
)
df = pd.read_csv(csv_path)


chemicals_df = df[df["category"] == "Chemicals & Reagents"]


def get_openai_key(env_file=None):
    env_path = Path(
        env_file or r"C:\LabGit\150citations classification\api_keys\OPEN_AI_KEY.env"
    )
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
    raise FileNotFoundError("OpenAI API key not found")


client = OpenAI(api_key=get_openai_key())


def reclassify_chemical(name):
    prompt = f"""
    You are a life-science cataloguing assistant.
    Reclassify the following chemical/reagent product into one of these subcategories and sub-subcategories:
    - Biochemical Reagents: Enzymes, Substrates, Dyes & Stains, Buffers
    - Small Molecules & Inhibitors: Chemical Compounds, Inhibitors, Antibiotics
    - Solvents & Solutions: Organic Solvents, Aqueous Solutions, Dilution Buffers

    If unsure, choose the most relevant subcategory and leave the sub-subcategory blank.
    Provide response ONLY as JSON, e.g., 
    {{"subcategory": "Biochemical Reagents", "sub_subcategory": "Enzymes"}}
    or if unsure:
    {{"subcategory": "Solvents & Solutions", "sub_subcategory": ""}}

    Product name: "{name}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        result = {"subcategory": "Solvents & Solutions", "sub_subcategory": ""}

    return result["subcategory"], result.get("sub_subcategory", "")


# Apply this to the chemicals_df
chemicals_df[["subcategory", "sub_subcategory"]] = chemicals_df["Name"].apply(
    lambda x: pd.Series(reclassify_chemical(x))
)

# Update the main DataFrame
df.update(chemicals_df)

df.to_csv("products_with_updated_solvents_categories.csv", index=False)
