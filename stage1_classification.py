import pandas as pd
import json
import logging
import os
import re
import yaml
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
from tqdm import tqdm
import glob
from pathlib import Path
from dotenv import load_dotenv
import random

# ─── CONFIG
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_with_biocomparecategories_n_confidence.csv"
VALIDATION_CSV = "validation_sample_100.csv"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 100
VALIDATION_SAMPLE_SIZE = 100
CATEGORY_YAML = "category_structure_copyofbiocompare.yaml"

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


class CategoryStructure:
    """Enhanced category structure handler for complex hierarchies"""

    def __init__(self, yaml_file: str = None):
        self.structure = {}
        self.flat_paths = (
            {}
        )  # For quick lookups: "category/subcategory/subsubcategory" -> depth
        self.max_depth = 0

        if yaml_file and os.path.exists(yaml_file):
            self.load_from_yaml(yaml_file)
        else:
            self.load_default_structure()

        self._build_flat_paths()

    def load_from_yaml(self, yaml_file: str):
        """Load category structure from YAML file"""
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.structure = data.get("categories", {})
            logger.info(f"Loaded category structure from {yaml_file}")
        except Exception as e:
            logger.error(f"Error loading YAML file {yaml_file}: {e}")
            self.load_default_structure()

    def load_default_structure(self):
        """Fallback to default structure if YAML fails"""
        self.structure = {
            "Antibodies": {
                "Primary Antibodies": [
                    "Monoclonal Antibodies",
                    "Polyclonal Antibodies",
                ],
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
            "Other": {},
        }

    def _build_flat_paths(self):
        """Build flat path structure for easy validation and lookup"""
        self.flat_paths = {}
        self.max_depth = 0

        def traverse(node, path, depth):
            self.max_depth = max(self.max_depth, depth)

            if isinstance(node, dict):
                # This is a category or subcategory with children
                for key, value in node.items():
                    new_path = f"{path}/{key}" if path else key
                    self.flat_paths[new_path] = depth + 1
                    traverse(value, new_path, depth + 1)
            elif isinstance(node, list):
                # This is a list of subcategories
                for item in node:
                    if isinstance(item, str):
                        item_path = f"{path}/{item}"
                        self.flat_paths[item_path] = depth + 1
                    elif isinstance(item, dict):
                        # Handle nested structures like "Fluorophore Conjugated Recombinant Antibodies"
                        for sub_key, sub_value in item.items():
                            sub_path = f"{path}/{sub_key}"
                            self.flat_paths[sub_path] = depth + 1
                            traverse(sub_value, sub_path, depth + 1)

        traverse(self.structure, "", 0)
        logger.info(
            f"Built flat paths with {len(self.flat_paths)} entries, max depth: {self.max_depth}"
        )

    def get_top_categories(self) -> List[str]:
        """Get list of top-level categories"""
        return list(self.structure.keys())

    def validate_path(
        self,
        category: str,
        subcategory: str = "",
        subsubcategory: str = "",
        subsubsubcategory: str = "",
    ) -> Tuple[bool, str]:
        """Validate a complete category path"""
        path_parts = [
            p for p in [category, subcategory, subsubcategory, subsubsubcategory] if p
        ]
        path = "/".join(path_parts)

        if path in self.flat_paths:
            return True, path

        # Try to find the longest valid path
        for i in range(len(path_parts) - 1, 0, -1):
            partial_path = "/".join(path_parts[:i])
            if partial_path in self.flat_paths:
                return True, partial_path

        return False, ""

    def get_same_top_level_categories(self, category: str) -> List[str]:
        """Get all subcategories under the same top-level category"""
        if category not in self.structure:
            return []

        all_paths = []
        top_structure = self.structure[category]

        def collect_paths(node, current_path):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = f"{current_path}/{key}" if current_path else key
                    all_paths.append(new_path)
                    collect_paths(value, new_path)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        all_paths.append(f"{current_path}/{item}")
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            sub_path = f"{current_path}/{sub_key}"
                            all_paths.append(sub_path)
                            collect_paths(sub_value, sub_path)

        collect_paths(top_structure, category)
        return all_paths

    def format_for_prompt(self) -> str:
        """Format the category structure for the LLM prompt"""

        def format_node(node, indent=0):
            result = []
            prefix = "  " * indent

            if isinstance(node, dict):
                for key, value in node.items():
                    result.append(f"{prefix}- {key}")
                    if value:  # Only recurse if there are children
                        result.extend(format_node(value, indent + 1))
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        result.append(f"{prefix}  • {item}")
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            result.append(f"{prefix}  • {sub_key}")
                            if sub_value:
                                result.extend(format_node(sub_value, indent + 2))

            return result

        lines = ["CATEGORY STRUCTURE:"]
        lines.extend(format_node(self.structure))
        return "\n".join(lines)


# Initialize category structure
cat_structure = CategoryStructure(CATEGORY_YAML)


def strip_code_fences(text: str) -> str:
    """Remove code fences from LLM response"""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def classify_product(name: str) -> Dict[str, str]:
    """Enhanced product classification with dual category support"""

    # Build prompt with category structure
    top_categories = cat_structure.get_top_categories()
    numbered_categories = "\n".join(
        f"{i+1}. {cat}" for i, cat in enumerate(top_categories)
    )
    formatted_structure = cat_structure.format_for_prompt()

    prompt = (
        "You are an expert life-science product cataloguing assistant.\n"
        "Given a product name, assign it to the most specific category possible from the hierarchical structure below.\n"
        "You can assign ONE primary classification and optionally ONE secondary classification if the product clearly fits two categories.\n"
        "IMPORTANT RULES:\n"
        "- Both primary and secondary classifications must be under the SAME top-level category\n"
        "- Only assign secondary classification if you have HIGH confidence for both\n"
        "- Provide confidence level as High, Medium, or Low\n"
        "- Leave fields empty if you cannot determine an appropriate classification at that level\n"
        "- Be conservative with dual classifications - only use when product genuinely serves both purposes\n\n"
        "Respond ONLY with raw JSON in this exact format:\n"
        '{"primary": {"category":"", "subcategory":"", "subsubcategory":"", "subsubsubcategory":"", "confidence":""}, "secondary": {"category":"", "subcategory":"", "subsubcategory":"", "subsubsubcategory":"", "confidence":""}, "has_secondary": false}\n\n'
        "Examples:\n"
        'Product: "Anti-Mouse IgG HRP Conjugated Secondary Antibody for Western Blot and ELISA"\n'
        'Response: {"primary": {"category":"Antibodies", "subcategory":"Secondary Antibodies", "subsubcategory":"Anti-Mouse Secondary Antibodies", "subsubsubcategory":"HRP Conjugated Secondary Antibodies", "confidence":"High"}, "secondary": {"category":"Antibodies", "subcategory":"Secondary Antibodies", "subsubcategory":"Western Blot Secondary Antibodies", "subsubsubcategory":"", "confidence":"High"}, "has_secondary": true}\n\n'
        'Product: "Human IL-6 ELISA Kit"\n'
        'Response: {"primary": {"category":"Assay Kits", "subcategory":"ELISA & Immunoassay Kits", "subsubcategory":"ELISA Kits", "subsubsubcategory":"", "confidence":"High"}, "secondary": {"category":"", "subcategory":"", "subsubcategory":"", "subsubsubcategory":"", "confidence":""}, "has_secondary": false}\n\n'
        'Product: "PCR Thermal Cycler"\n'
        'Response: {"primary": {"category":"Lab Equipment", "subcategory":"General Lab Equipment", "subsubcategory":"Thermal Cyclers", "subsubsubcategory":"", "confidence":"High"}, "secondary": {"category":"", "subcategory":"", "subsubcategory":"", "subsubsubcategory":"", "confidence":""}, "has_secondary": false}\n\n'
        f"Valid top-level categories:\n{numbered_categories}\n\n"
        f"{formatted_structure}\n\n"
        f'Product name: "{name}"\n'
        "Response:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more consistent results
        )
        text = resp.choices[0].message.content.strip()
        stripped = strip_code_fences(text)

        j = json.loads(stripped)

        # Extract primary classification
        primary = j.get("primary", {})
        secondary = j.get("secondary", {})
        has_secondary = j.get("has_secondary", False)

        # Validate primary classification
        primary_category = primary.get("category", "Other")
        primary_subcategory = primary.get("subcategory", "")
        primary_subsubcategory = primary.get("subsubcategory", "")
        primary_subsubsubcategory = primary.get("subsubsubcategory", "")
        primary_confidence = primary.get("confidence", "Low")

        # Validate primary path
        is_valid_primary, valid_primary_path = cat_structure.validate_path(
            primary_category,
            primary_subcategory,
            primary_subsubcategory,
            primary_subsubsubcategory,
        )

        if not is_valid_primary:
            logger.warning(
                f"Invalid primary classification for '{name}': {primary_category}/{primary_subcategory}/{primary_subsubcategory}/{primary_subsubsubcategory}"
            )
            (
                primary_category,
                primary_subcategory,
                primary_subsubcategory,
                primary_subsubsubcategory,
            ) = ("Other", "", "", "")
            primary_confidence = "Low"
            has_secondary = False

        # Initialize secondary fields
        secondary_category = ""
        secondary_subcategory = ""
        secondary_subsubcategory = ""
        secondary_subsubsubcategory = ""
        secondary_confidence = ""

        # Validate secondary classification if it exists
        if has_secondary and secondary:
            secondary_category = secondary.get("category", "")
            secondary_subcategory = secondary.get("subcategory", "")
            secondary_subsubcategory = secondary.get("subsubcategory", "")
            secondary_subsubsubcategory = secondary.get("subsubsubcategory", "")
            secondary_confidence = secondary.get("confidence", "")

            # Check if secondary is under same top-level category
            if secondary_category != primary_category:
                logger.warning(
                    f"Secondary category must be under same top-level as primary for '{name}'"
                )
                has_secondary = False
                secondary_category = secondary_subcategory = (
                    secondary_subsubcategory
                ) = secondary_subsubsubcategory = secondary_confidence = ""
            else:
                # Validate secondary path
                is_valid_secondary, valid_secondary_path = cat_structure.validate_path(
                    secondary_category,
                    secondary_subcategory,
                    secondary_subsubcategory,
                    secondary_subsubsubcategory,
                )

                if not is_valid_secondary:
                    logger.warning(
                        f"Invalid secondary classification for '{name}': {secondary_category}/{secondary_subcategory}/{secondary_subsubcategory}/{secondary_subsubsubcategory}"
                    )
                    has_secondary = False
                    secondary_category = secondary_subcategory = (
                        secondary_subsubcategory
                    ) = secondary_subsubsubcategory = secondary_confidence = ""

                # Check if both primary and secondary have High confidence for dual classification
                if primary_confidence != "High" or secondary_confidence != "High":
                    logger.info(
                        f"Dual classification requires High confidence for both categories for '{name}'"
                    )
                    has_secondary = False
                    secondary_category = secondary_subcategory = (
                        secondary_subsubcategory
                    ) = secondary_subsubsubcategory = secondary_confidence = ""

        return {
            "category": primary_category,
            "subcategory": primary_subcategory,
            "subsubcategory": primary_subsubcategory,
            "subsubsubcategory": primary_subsubsubcategory,
            "confidence": primary_confidence,
            "secondary_category": secondary_category,
            "secondary_subcategory": secondary_subcategory,
            "secondary_subsubcategory": secondary_subsubcategory,
            "secondary_subsubsubcategory": secondary_subsubsubcategory,
            "secondary_confidence": secondary_confidence,
            "has_secondary": has_secondary,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for '{name}': {stripped!r} - {e}")
        return {
            "category": "Other",
            "subcategory": "",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "secondary_category": "",
            "secondary_subcategory": "",
            "secondary_subsubcategory": "",
            "secondary_subsubsubcategory": "",
            "secondary_confidence": "",
            "has_secondary": False,
        }
    except Exception as e:
        logger.error(f"Error classifying '{name}': {e}")
        return {
            "category": "Other",
            "subcategory": "",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "secondary_category": "",
            "secondary_subcategory": "",
            "secondary_subsubcategory": "",
            "secondary_subsubsubcategory": "",
            "secondary_confidence": "",
            "has_secondary": False,
        }


def create_validation_sample(
    input_csv: str, sample_size: int = VALIDATION_SAMPLE_SIZE
) -> pd.DataFrame:
    """Create a random sample for manual validation"""
    try:
        df = pd.read_csv(input_csv)

        if len(df) < sample_size:
            logger.warning(
                f"Dataset has only {len(df)} products, using all for validation"
            )
            sample_df = df.copy()
        else:
            # Create random sample
            sample_indices = random.sample(range(len(df)), sample_size)
            sample_df = df.iloc[sample_indices].copy()

        # Initialize classification columns
        sample_df["category"] = ""
        sample_df["subcategory"] = ""
        sample_df["subsubcategory"] = ""
        sample_df["subsubsubcategory"] = ""
        sample_df["confidence"] = ""
        sample_df["secondary_category"] = ""
        sample_df["secondary_subcategory"] = ""
        sample_df["secondary_subsubcategory"] = ""
        sample_df["secondary_subsubsubcategory"] = ""
        sample_df["secondary_confidence"] = ""
        sample_df["has_secondary"] = False
        sample_df["needs_stage2"] = False  # Will be populated during classification

        logger.info(f"Created validation sample with {len(sample_df)} products")
        return sample_df

    except Exception as e:
        logger.error(f"Error creating validation sample: {e}")
        raise


def process_validation_sample():
    """Process the validation sample of 100 products"""
    logger.info("Starting validation sample processing...")

    # Check if validation sample already exists
    if os.path.exists(VALIDATION_CSV):
        response = input(
            f"Validation file {VALIDATION_CSV} already exists. Do you want to:\n1. Load existing file\n2. Create new sample\nEnter choice (1 or 2): "
        )
        if response == "1":
            validation_df = pd.read_csv(VALIDATION_CSV)
            logger.info(
                f"Loaded existing validation sample with {len(validation_df)} products"
            )
        else:
            validation_df = create_validation_sample(INPUT_CSV)
    else:
        validation_df = create_validation_sample(INPUT_CSV)

    # Check if already processed
    if (
        validation_df["category"].notna().any()
        and validation_df["category"].ne("").any()
    ):
        response = input(
            "Validation sample appears to be already processed. Do you want to:\n1. Use existing results\n2. Reprocess all\n3. Continue from where left off\nEnter choice (1, 2, or 3): "
        )
        if response == "1":
            logger.info("Using existing validation results")
        elif response == "2":
            # Reset all classification columns
            validation_df["category"] = ""
            validation_df["subcategory"] = ""
            validation_df["subsubcategory"] = ""
            validation_df["subsubsubcategory"] = ""
            validation_df["confidence"] = ""
            validation_df["secondary_category"] = ""
            validation_df["secondary_subcategory"] = ""
            validation_df["secondary_subsubcategory"] = ""
            validation_df["secondary_subsubsubcategory"] = ""
            validation_df["secondary_confidence"] = ""
            validation_df["has_secondary"] = False
            validation_df["needs_stage2"] = False
        # For choice 3, we continue from where we left off

    # Process unclassified products
    unprocessed_mask = validation_df["category"].isna() | (
        validation_df["category"] == ""
    )
    unprocessed_indices = validation_df[unprocessed_mask].index

    if len(unprocessed_indices) > 0:
        logger.info(
            f"Processing {len(unprocessed_indices)} unclassified products in validation sample..."
        )

        for idx in tqdm(unprocessed_indices, desc="Validating products"):
            name = validation_df.at[idx, "Name"]
            result = classify_product(name)

            # Update dataframe
            validation_df.at[idx, "category"] = result["category"]
            validation_df.at[idx, "subcategory"] = result["subcategory"]
            validation_df.at[idx, "subsubcategory"] = result["subsubcategory"]
            validation_df.at[idx, "subsubsubcategory"] = result["subsubsubcategory"]
            validation_df.at[idx, "confidence"] = result["confidence"]
            validation_df.at[idx, "secondary_category"] = result["secondary_category"]
            validation_df.at[idx, "secondary_subcategory"] = result[
                "secondary_subcategory"
            ]
            validation_df.at[idx, "secondary_subsubcategory"] = result[
                "secondary_subsubcategory"
            ]
            validation_df.at[idx, "secondary_subsubsubcategory"] = result[
                "secondary_subsubsubcategory"
            ]
            validation_df.at[idx, "secondary_confidence"] = result[
                "secondary_confidence"
            ]
            validation_df.at[idx, "has_secondary"] = result["has_secondary"]

            # Mark for Stage 2 if needed
            needs_stage2 = (
                result["confidence"] == "Low" or result["category"] == "Other"
            )
            validation_df.at[idx, "needs_stage2"] = needs_stage2

    # Save validation results
    validation_df.to_csv(VALIDATION_CSV, index=False)
    logger.info(f"Validation sample saved to {VALIDATION_CSV}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SAMPLE SUMMARY")
    print("=" * 60)
    print(f"Total products validated: {len(validation_df)}")
    print(f"Category distribution:")
    print(validation_df["category"].value_counts())
    print(f"\nConfidence distribution:")
    print(validation_df["confidence"].value_counts())
    print(f"\nDual classifications: {validation_df['has_secondary'].sum()}")
    print(f"Products needing Stage 2: {validation_df['needs_stage2'].sum()}")

    # Ask for user decision
    print("\n" + "=" * 60)
    print("MANUAL REVIEW REQUIRED")
    print("=" * 60)
    print(f"Please manually review the file: {VALIDATION_CSV}")
    print("Check the accuracy of classifications, especially:")
    print("- Category assignments")
    print("- Confidence levels")
    print("- Dual classifications")
    print("\nAfter review, you can:")
    print("1. Proceed with full dataset if accuracy >= 90%")
    print("2. Adjust prompts and retry if accuracy < 90%")

    return validation_df


def process_full_dataset():
    """Process the full dataset after validation approval"""

    # Find the latest checkpoint
    CHECKPOINT_FILE = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.csv"))
        if checkpoint_files:
            # Get the checkpoint file with the highest number
            checkpoint_numbers = []
            for cp_file in checkpoint_files:
                match = re.search(r"checkpoint_(\d+)\.csv", cp_file)
                if match:
                    checkpoint_numbers.append((int(match.group(1)), cp_file))

            if checkpoint_numbers:
                checkpoint_numbers.sort(reverse=True)
                CHECKPOINT_FILE = checkpoint_numbers[0][1]

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load or create dataframe
    if CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        # Find the first row with empty 'category' to resume from there
        try:
            empty_mask = df["category"].isna() | (df["category"] == "")
            if empty_mask.any():
                start_idx = df[empty_mask].index[0]
            else:
                logger.info("Checkpoint file is fully processed. Exiting.")
                return
        except (KeyError, IndexError):
            start_idx = 0
        logger.info(f"Resuming from checkpoint {CHECKPOINT_FILE} at index {start_idx}")
    else:
        df = pd.read_csv(INPUT_CSV)
        # Initialize classification columns
        df["category"] = ""
        df["subcategory"] = ""
        df["subsubcategory"] = ""
        df["subsubsubcategory"] = ""
        df["confidence"] = ""
        df["secondary_category"] = ""
        df["secondary_subcategory"] = ""
        df["secondary_subsubcategory"] = ""
        df["secondary_subsubsubcategory"] = ""
        df["secondary_confidence"] = ""
        df["has_secondary"] = False
        df["needs_stage2"] = False
        start_idx = 0
        logger.info("No checkpoint found, starting classification from beginning.")

    total = len(df)
    logger.info(f"Total products to classify: {total}")
    logger.info(f"Starting from index: {start_idx}")

    # Process products
    stage2_count = 0
    dual_classification_count = 0

    for idx in tqdm(range(start_idx, total), desc="Classifying products"):
        name = df.at[idx, "Name"]
        result = classify_product(name)

        # Update dataframe
        df.at[idx, "category"] = result["category"]
        df.at[idx, "subcategory"] = result["subcategory"]
        df.at[idx, "subsubcategory"] = result["subsubcategory"]
        df.at[idx, "subsubsubcategory"] = result["subsubsubcategory"]
        df.at[idx, "confidence"] = result["confidence"]
        df.at[idx, "secondary_category"] = result["secondary_category"]
        df.at[idx, "secondary_subcategory"] = result["secondary_subcategory"]
        df.at[idx, "secondary_subsubcategory"] = result["secondary_subsubcategory"]
        df.at[idx, "secondary_subsubsubcategory"] = result[
            "secondary_subsubsubcategory"
        ]
        df.at[idx, "secondary_confidence"] = result["secondary_confidence"]
        df.at[idx, "has_secondary"] = result["has_secondary"]

        # Mark for Stage 2 if needed
        needs_stage2 = result["confidence"] == "Low" or result["category"] == "Other"
        df.at[idx, "needs_stage2"] = needs_stage2

        if needs_stage2:
            stage2_count += 1
        if result["has_secondary"]:
            dual_classification_count += 1

        # Save checkpoint
        if (idx + 1) % CHECKPOINT_FREQ == 0 or idx == total - 1:
            cp = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx+1}.csv")
            df.to_csv(cp, index=False)
            logger.info(
                f"Saved checkpoint: {cp} | Stage 2 needed: {stage2_count}/{idx+1} ({stage2_count/(idx+1)*100:.1f}%)"
            )

    # Save final results
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Classification complete. Results saved to {OUTPUT_CSV}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("FULL DATASET CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total products classified: {len(df)}")
    print(f"Top-level category distribution:")
    print(df["category"].value_counts())
    print(f"\nConfidence distribution:")
    print(df["confidence"].value_counts())
    print(
        f"\nDual classifications: {dual_classification_count} ({dual_classification_count/len(df)*100:.1f}%)"
    )
    print(f"Products needing Stage 2: {stage2_count} ({stage2_count/len(df)*100:.1f}%)")


def main():
    """Main execution function"""

    print("=" * 60)
    print("STAGE 1: PRODUCT CLASSIFICATION SYSTEM")
    print("=" * 60)

    # Check if category YAML file exists
    if not os.path.exists(CATEGORY_YAML):
        logger.error(f"Category YAML file not found: {CATEGORY_YAML}")
        logger.info("Please ensure the YAML file is in the correct location")
        return

    print(f"Loaded category structure from: {CATEGORY_YAML}")
    print(f"Total categories found: {len(cat_structure.get_top_categories())}")
    print(f"Maximum hierarchy depth: {cat_structure.max_depth}")

    # Step 1: Process validation sample
    print("\nStep 1: Processing validation sample...")
    validation_df = process_validation_sample()

    # Wait for user approval
    print("\n" + "=" * 60)
    user_input = input(
        "After reviewing the validation results, do you want to proceed with the full dataset? (y/n): "
    )

    if user_input.lower() != "y":
        print("Stopping processing. You can:")
        print("1. Adjust prompts in the classify_product() function")
        print("2. Modify the category structure YAML file")
        print("3. Re-run the validation sample")
        return

    # Step 2: Process full dataset
    print("\nStep 2: Processing full dataset...")
    process_full_dataset()

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Review the final results in:", OUTPUT_CSV)
    print("2. Products marked 'needs_stage2=True' are ready for Stage 2 processing")
    print("3. Dual classifications are marked with 'has_secondary=True'")
    print("4. All checkpoints are saved in:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
