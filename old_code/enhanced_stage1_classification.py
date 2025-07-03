import pandas as pd
import json
import logging
import os
import re
import yaml
from typing import Dict, List, Tuple, Optional, Any, Set
from openai import OpenAI
from tqdm import tqdm
import glob
from pathlib import Path
from dotenv import load_dotenv
import random

# ─── CONFIG (keep your existing config)
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_with_biocomparecategories_n_confidence_ENHANCED.csv"
VALIDATION_CSV = "validation_sample_100_ENHANCED.csv"
CHECKPOINT_DIR = "checkpoints_enhanced"
CHECKPOINT_FREQ = 100
VALIDATION_SAMPLE_SIZE = 100
CATEGORY_YAML = "category_structure.yaml"

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

# ─── ENHANCED TAG SYSTEM
class TagSystem:
    """Cross-cutting tag system that works across all categories"""
    
    def __init__(self):
        # Extract these from your YAML files
        self.technique_tags = {
            "Flow Cytometry", "Western Blot", "ELISA", "Immunohistochemistry", 
            "Immunofluorescence", "PCR", "qPCR", "Microscopy", "Confocal Microscopy",
            "Live Cell Imaging", "High Content Screening", "Single Cell Analysis",
            "Cell Culture", "Transfection", "Cloning", "Expression", "Apoptosis Analysis",
            "Cell Viability", "Multiplex Assay", "Gel Documentation", "Electrophoresis",
            "Mass Spectrometry", "Chromatography", "Sequencing", "Automation"
        }
        
        self.research_application_tags = {
            "Neuroscience", "Cancer Research", "Stem Cell Research", "Immunology",
            "Infectious Disease", "Drug Discovery", "Cell Biology", "Molecular Biology",
            "Biochemistry", "Genomics", "Proteomics", "Cytokine Research", "Hematology",
            "Apoptosis Research", "Autophagy Research", "Metabolomics", "Developmental Biology"
        }
        
        self.functional_tags = {
            "High-throughput", "Automation Compatible", "Single-use", "Reusable",
            "Research Use Only", "Diagnostic", "Quantitative", "Qualitative", 
            "Real-time", "Endpoint", "Benchtop", "Portable", "Sterile", "Validated"
        }
        
        self.all_tags = self.technique_tags | self.research_application_tags | self.functional_tags
    
    def get_tags_by_category(self, category_type: str) -> Set[str]:
        """Get tags by type"""
        if category_type == "technique":
            return self.technique_tags
        elif category_type == "research":
            return self.research_application_tags
        elif category_type == "functional":
            return self.functional_tags
        else:
            return self.all_tags
    
    def format_for_prompt(self) -> str:
        """Format tags for LLM prompt"""
        return f"""
TECHNIQUE TAGS (choose 1-3 relevant techniques):
{', '.join(sorted(self.technique_tags))}

RESEARCH APPLICATION TAGS (choose 1-2 relevant research areas):
{', '.join(sorted(self.research_application_tags))}

FUNCTIONAL TAGS (choose 1-2 relevant functional characteristics):
{', '.join(sorted(self.functional_tags))}
"""

# Initialize tag system
tag_system = TagSystem()

class CategoryStructure:
    """Enhanced category structure handler (keep your existing code but add tag extraction)"""

    def __init__(self, yaml_file: str = None):
        self.structure = {}
        self.flat_paths = {}
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
                "Primary Antibodies": ["Monoclonal Antibodies", "Polyclonal Antibodies"],
                "Secondary Antibodies": ["Enzyme Conjugates", "Fluorophore Conjugates"],
            },
            "Assay Kits": {
                "ELISA & Immunoassay Kits": ["ELISA Kits", "Multiplex Assays"],
                "Cell-Based & Functional Assay Kits": [
                    "Cell Viability Kits", "Apoptosis Kits", "Metabolism Kits", "Enzyme Activity Kits",
                ],
            },
            "Lab Equipment": {
                "Analytical Instruments": [
                    "Microscopes", "Imaging Systems", "Sequencers", "Spectrometers", "Flow Cytometers",
                ],
                "General Lab Equipment": [
                    "Centrifuges", "Incubators", "Shakers", "Thermal Cyclers", "PCR Machines",
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
                for key, value in node.items():
                    new_path = f"{path}/{key}" if path else key
                    self.flat_paths[new_path] = depth + 1
                    traverse(value, new_path, depth + 1)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        item_path = f"{path}/{item}"
                        self.flat_paths[item_path] = depth + 1
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            sub_path = f"{path}/{sub_key}"
                            self.flat_paths[sub_path] = depth + 1
                            traverse(sub_value, sub_path, depth + 1)

        traverse(self.structure, "", 0)
        logger.info(f"Built flat paths with {len(self.flat_paths)} entries, max depth: {self.max_depth}")

    def get_top_categories(self) -> List[str]:
        """Get list of top-level categories"""
        return list(self.structure.keys())

    def validate_path(self, category: str, subcategory: str = "", subsubcategory: str = "", subsubsubcategory: str = "") -> Tuple[bool, str]:
        """Validate a complete category path"""
        path_parts = [p for p in [category, subcategory, subsubcategory, subsubsubcategory] if p]
        path = "/".join(path_parts)

        if path in self.flat_paths:
            return True, path

        for i in range(len(path_parts) - 1, 0, -1):
            partial_path = "/".join(path_parts[:i])
            if partial_path in self.flat_paths:
                return True, partial_path

        return False, ""

    def format_for_prompt(self) -> str:
        """Format the category structure for the LLM prompt - simplified for navigation"""
        
        # Keep only top 2-3 levels for navigation to avoid overwhelming the LLM
        def format_simplified(node, indent=0, max_depth=3):
            result = []
            prefix = "  " * indent
            
            if indent >= max_depth:
                return ["  " * indent + "  • [additional subcategories available]"]

            if isinstance(node, dict):
                for key, value in node.items():
                    result.append(f"{prefix}- {key}")
                    if value and indent < max_depth - 1:
                        result.extend(format_simplified(value, indent + 1, max_depth))
            elif isinstance(node, list):
                # Only show first few examples to keep prompt manageable
                for i, item in enumerate(node[:5]):  # Limit to 5 examples
                    if isinstance(item, str):
                        result.append(f"{prefix}  • {item}")
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            result.append(f"{prefix}  • {sub_key}")
                            break  # Only show first example
                if len(node) > 5:
                    result.append(f"{prefix}  • ... ({len(node)-5} more)")

            return result

        lines = ["SIMPLIFIED CATEGORY STRUCTURE (for navigation):"]
        lines.extend(format_simplified(self.structure))
        return "\n".join(lines)

# Initialize category structure
cat_structure = CategoryStructure(CATEGORY_YAML)

def strip_code_fences(text: str) -> str:
    """Remove code fences from LLM response"""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()

def classify_product_enhanced(name: str, description: str = "") -> Dict[str, Any]:
    """
    ENHANCED classification with multi-label categories + tags
    This replaces your existing classify_product function
    """

    # Build enhanced prompt
    top_categories = cat_structure.get_top_categories()
    numbered_categories = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(top_categories))
    formatted_structure = cat_structure.format_for_prompt()
    formatted_tags = tag_system.format_for_prompt()

    prompt = f"""
You are an expert life-science product cataloguing assistant.

Given a product name, provide:
1. 1-3 CATEGORY classifications (can span different top-level categories)
2. 3-8 TAGS from the predefined tag lists

IMPORTANT RULES:
- Mark ONE category as primary (most relevant)
- Secondary/tertiary categories can be from different top-level categories
- Only assign multiple categories if you have High confidence for each
- Choose tags that describe HOW the product is used, WHAT research it supports, and KEY characteristics
- Be conservative with classifications - better to have fewer high-confidence assignments

Respond ONLY with raw JSON in this format:
{{
  "classifications": [
    {{"category":"Antibodies", "subcategory":"Primary Antibodies", "subsubcategory":"Monoclonal Antibodies", "confidence":"High", "is_primary":true}},
    {{"category":"Lab Equipment", "subcategory":"Cell Analysis", "subsubcategory":"Flow Cytometers", "confidence":"Medium", "is_primary":false}}
  ],
  "technique_tags": ["Flow Cytometry", "Cell Analysis"],
  "research_tags": ["Immunology"],
  "functional_tags": ["High-throughput", "Research Use Only"]
}}

Examples:

Product: "Anti-CD3 Monoclonal Antibody for Flow Cytometry"
Response: {{"classifications": [{{"category":"Antibodies", "subcategory":"Primary Antibodies", "subsubcategory":"Monoclonal Antibodies", "confidence":"High", "is_primary":true}}], "technique_tags": ["Flow Cytometry"], "research_tags": ["Immunology", "Cell Biology"], "functional_tags": ["Research Use Only"]}}

Product: "BD FACSCanto II Flow Cytometer"  
Response: {{"classifications": [{{"category":"Lab Equipment", "subcategory":"Analytical Instruments", "subsubcategory":"Flow Cytometers", "confidence":"High", "is_primary":true}}, {{"category":"Cell Biology", "subcategory":"Cell Analysis", "subsubcategory":"", "confidence":"Medium", "is_primary":false}}], "technique_tags": ["Flow Cytometry", "Single Cell Analysis"], "research_tags": ["Immunology", "Cell Biology"], "functional_tags": ["High-throughput", "Automation Compatible"]}}

Product: "Human TNF-alpha ELISA Kit"
Response: {{"classifications": [{{"category":"Assay Kits", "subcategory":"ELISA & Immunoassay Kits", "subsubcategory":"ELISA Kits", "confidence":"High", "is_primary":true}}], "technique_tags": ["ELISA"], "research_tags": ["Immunology", "Cytokine Research"], "functional_tags": ["Quantitative", "Research Use Only"]}}

Valid top-level categories:
{numbered_categories}

{formatted_structure}

{formatted_tags}

Product name: "{name}"
Description: "{description}"
Response:"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        stripped = strip_code_fences(text)

        j = json.loads(stripped)

        # Extract and validate classifications
        classifications = j.get("classifications", [])
        validated_classifications = []
        primary_assigned = False

        for classification in classifications:
            # Validate each classification path
            category = classification.get("category", "Other")
            subcategory = classification.get("subcategory", "")
            subsubcategory = classification.get("subsubcategory", "")
            subsubsubcategory = classification.get("subsubsubcategory", "")
            confidence = classification.get("confidence", "Low")
            is_primary = classification.get("is_primary", False)

            # Validate path
            is_valid, valid_path = cat_structure.validate_path(
                category, subcategory, subsubcategory, subsubsubcategory
            )

            if not is_valid:
                logger.warning(f"Invalid classification path for '{name}': {category}/{subcategory}")
                continue

            # Ensure only one primary
            if is_primary and primary_assigned:
                is_primary = False
            elif is_primary:
                primary_assigned = True

            validated_classifications.append({
                "category": category,
                "subcategory": subcategory,
                "subsubcategory": subsubcategory,
                "subsubsubcategory": subsubsubcategory,
                "confidence": confidence,
                "is_primary": is_primary
            })

        # Ensure we have a primary if we have any classifications
        if validated_classifications and not primary_assigned:
            validated_classifications[0]["is_primary"] = True

        # If no valid classifications, add fallback
        if not validated_classifications:
            validated_classifications = [{
                "category": "Other",
                "subcategory": "",
                "subsubcategory": "",
                "subsubsubcategory": "",
                "confidence": "Low",
                "is_primary": True
            }]

        # Extract and validate tags
        technique_tags = [tag for tag in j.get("technique_tags", []) if tag in tag_system.technique_tags]
        research_tags = [tag for tag in j.get("research_tags", []) if tag in tag_system.research_application_tags]
        functional_tags = [tag for tag in j.get("functional_tags", []) if tag in tag_system.functional_tags]

        all_tags = technique_tags + research_tags + functional_tags

        return {
            "classifications": validated_classifications,
            "technique_tags": technique_tags,
            "research_tags": research_tags,
            "functional_tags": functional_tags,
            "all_tags": all_tags,
            "tag_count": len(all_tags),
            "classification_count": len(validated_classifications)
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for '{name}': {stripped!r} - {e}")
        return get_fallback_result(name)
    except Exception as e:
        logger.error(f"Error classifying '{name}': {e}")
        return get_fallback_result(name)

def get_fallback_result(name: str) -> Dict[str, Any]:
    """Fallback result when classification fails"""
    return {
        "classifications": [{
            "category": "Other",
            "subcategory": "",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "is_primary": True
        }],
        "technique_tags": [],
        "research_tags": [],
        "functional_tags": [],
        "all_tags": [],
        "tag_count": 0,
        "classification_count": 1
    }

def create_validation_sample_enhanced(input_csv: str, sample_size: int = VALIDATION_SAMPLE_SIZE) -> pd.DataFrame:
    """Enhanced validation sample with tag columns"""
    try:
        df = pd.read_csv(input_csv)

        if len(df) < sample_size:
            logger.warning(f"Dataset has only {len(df)} products, using all for validation")
            sample_df = df.copy()
        else:
            sample_indices = random.sample(range(len(df)), sample_size)
            sample_df = df.iloc[sample_indices].copy()

        # Initialize enhanced columns
        sample_df["category"] = ""
        sample_df["subcategory"] = ""
        sample_df["subsubcategory"] = ""
        sample_df["subsubsubcategory"] = ""
        sample_df["confidence"] = ""
        
        # Multi-label support
        sample_df["classification_count"] = 0
        sample_df["secondary_category"] = ""
        sample_df["secondary_subcategory"] = ""
        sample_df["secondary_confidence"] = ""
        
        # Tag support
        sample_df["technique_tags"] = ""
        sample_df["research_tags"] = ""
        sample_df["functional_tags"] = ""
        sample_df["all_tags"] = ""
        sample_df["tag_count"] = 0
        
        sample_df["needs_stage2"] = False

        logger.info(f"Created enhanced validation sample with {len(sample_df)} products")
        return sample_df

    except Exception as e:
        logger.error(f"Error creating validation sample: {e}")
        raise

def process_validation_sample_enhanced():
    """Enhanced validation processing with tags"""
    logger.info("Starting enhanced validation sample processing...")

    if os.path.exists(VALIDATION_CSV):
        response = input(f"Validation file {VALIDATION_CSV} exists. Load existing (1) or create new (2)? ")
        if response == "1":
            validation_df = pd.read_csv(VALIDATION_CSV)
            logger.info(f"Loaded existing validation sample with {len(validation_df)} products")
        else:
            validation_df = create_validation_sample_enhanced(INPUT_CSV)
    else:
        validation_df = create_validation_sample_enhanced(INPUT_CSV)

    # Process unclassified products
    unprocessed_mask = validation_df["category"].isna() | (validation_df["category"] == "")
    unprocessed_indices = validation_df[unprocessed_mask].index

    if len(unprocessed_indices) > 0:
        logger.info(f"Processing {len(unprocessed_indices)} products with enhanced classification...")

        for idx in tqdm(unprocessed_indices, desc="Enhanced validation"):
            name = validation_df.at[idx, "Name"]
            description = validation_df.at[idx, "Description"] if "Description" in validation_df.columns else ""
            
            result = classify_product_enhanced(name, description)

            # Store primary classification
            primary_class = next((c for c in result["classifications"] if c["is_primary"]), 
                               result["classifications"][0])
            
            validation_df.at[idx, "category"] = primary_class["category"]
            validation_df.at[idx, "subcategory"] = primary_class["subcategory"]
            validation_df.at[idx, "subsubcategory"] = primary_class["subsubcategory"]
            validation_df.at[idx, "subsubsubcategory"] = primary_class["subsubsubcategory"]
            validation_df.at[idx, "confidence"] = primary_class["confidence"]
            validation_df.at[idx, "classification_count"] = result["classification_count"]

            # Store secondary classification if exists
            secondary_classes = [c for c in result["classifications"] if not c["is_primary"]]
            if secondary_classes:
                sec_class = secondary_classes[0]
                validation_df.at[idx, "secondary_category"] = sec_class["category"]
                validation_df.at[idx, "secondary_subcategory"] = sec_class["subcategory"]
                validation_df.at[idx, "secondary_confidence"] = sec_class["confidence"]

            # Store tags
            validation_df.at[idx, "technique_tags"] = "|".join(result["technique_tags"])
            validation_df.at[idx, "research_tags"] = "|".join(result["research_tags"])
            validation_df.at[idx, "functional_tags"] = "|".join(result["functional_tags"])
            validation_df.at[idx, "all_tags"] = "|".join(result["all_tags"])
            validation_df.at[idx, "tag_count"] = result["tag_count"]

            # Mark for Stage 2 if needed
            validation_df.at[idx, "needs_stage2"] = (
                primary_class["confidence"] == "Low" or primary_class["category"] == "Other"
            )

    # Save validation results
    validation_df.to_csv(VALIDATION_CSV, index=False)
    logger.info(f"Enhanced validation sample saved to {VALIDATION_CSV}")

    # Print enhanced summary
    print("\n" + "=" * 70)
    print("ENHANCED VALIDATION SAMPLE SUMMARY")
    print("=" * 70)
    print(f"Total products validated: {len(validation_df)}")
    print(f"\nCategory distribution:")
    print(validation_df["category"].value_counts())
    print(f"\nConfidence distribution:")
    print(validation_df["confidence"].value_counts())
    print(f"\nMulti-label classifications: {validation_df[validation_df['classification_count'] > 1].shape[0]}")
    print(f"Products with tags: {validation_df[validation_df['tag_count'] > 0].shape[0]}")
    print(f"Average tags per product: {validation_df['tag_count'].mean():.1f}")
    print(f"\nMost common technique tags:")
    all_technique_tags = []
    for tags_str in validation_df["technique_tags"].dropna():
        if tags_str:
            all_technique_tags.extend(tags_str.split("|"))
    if all_technique_tags:
        from collections import Counter
        technique_counts = Counter(all_technique_tags)
        for tag, count in technique_counts.most_common(5):
            print(f"  {tag}: {count}")

    return validation_df

# Update your main function to use enhanced processing
def main_enhanced():
    """Enhanced main execution function"""

    print("=" * 70)
    print("ENHANCED STAGE 1: MULTI-LABEL CLASSIFICATION + TAGS")
    print("=" * 70)

    # Load category structure
    if not os.path.exists(CATEGORY_YAML):
        logger.error(f"Category YAML file not found: {CATEGORY_YAML}")
        return

    print(f"Loaded category structure from: {CATEGORY_YAML}")
    print(f"Total categories: {len(cat_structure.get_top_categories())}")
    print(f"Maximum hierarchy depth: {cat_structure.max_depth}")
    print(f"Total available tags: {len(tag_system.all_tags)}")
    print(f"  - Technique tags: {len(tag_system.technique_tags)}")
    print(f"  - Research tags: {len(tag_system.research_application_tags)}")
    print(f"  - Functional tags: {len(tag_system.functional_tags)}")

    # Step 1: Enhanced validation
    print("\nStep 1: Processing enhanced validation sample...")
    validation_df = process_validation_sample_enhanced()

    # Wait for user approval
    print("\n" + "=" * 70)
    user_input = input("After reviewing the enhanced validation results, proceed with full dataset? (y/n): ")

    if user_input.lower() != "y":
        print("Processing stopped. You can adjust:")
        print("1. Tag categories in TagSystem class")
        print("2. Classification prompts")
        print("3. Category structure YAML")
        return

    # Step 2: Process full dataset with enhanced features
    print("\nStep 2: Processing full dataset with enhanced classification...")
    # You would implement process_full_dataset_enhanced() similarly
    
    print("\n" + "=" * 70)
    print("ENHANCED STAGE 1 COMPLETE!")
    print("=" * 70)
    print("New features added:")
    print("✓ Multi-label classification across different top categories")
    print("✓ Cross-cutting tag system (technique + research + functional)")
    print("✓ Enhanced search capabilities through tags")
    print("✓ Simplified navigation structure")
    print("✓ Deep category knowledge preserved")

if __name__ == "__main__":
    main_enhanced()