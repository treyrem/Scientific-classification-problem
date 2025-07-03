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
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_with_enhanced_multilabel_classification.csv"
VALIDATION_CSV = "validation_sample_enhanced_multilabel.csv"
CHECKPOINT_DIR = "checkpoints_enhanced_multilabel"
CHECKPOINT_FREQ = 100
VALIDATION_SAMPLE_SIZE = 100

# YAML file configuration - FIXED PATHS
YAML_DIRECTORY = "C:/LabGit/150citations classification/category_structures"  # Directory containing all YAML files
MASTER_CATEGORIES_FILE = "C:/LabGit/150citations classification/master_categories.yaml"  # High-level category mapping

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED TAG SYSTEM WITH COMPREHENSIVE LIFE SCIENCE COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════


class EnhancedTagSystem:
    """Comprehensive cross-cutting tag system for life science products"""

    def __init__(self):
        # Core technique/application tags
        self.technique_tags = {
            # Antibody Techniques
            "Western Blot",
            "Immunohistochemistry",
            "Immunofluorescence",
            "Flow Cytometry",
            "ELISA",
            "Immunoprecipitation",
            "ChIP",
            "ELISPOT",
            "Immunocytochemistry",
            # Cell Analysis
            "Cell Culture",
            "Live Cell Imaging",
            "High Content Screening",
            "Single Cell Analysis",
            "Cell Viability",
            "Apoptosis Analysis",
            "Cell Cycle Analysis",
            "Cell Tracking",
            "Cell Sorting",
            "Cell Counting",
            "Migration Assay",
            "Invasion Assay",
            # Molecular Biology
            "PCR",
            "qPCR",
            "RT-PCR",
            "DNA Sequencing",
            "RNA Sequencing",
            "Cloning",
            "Transfection",
            "Transformation",
            "Gene Expression",
            "In Situ Hybridization",
            "Northern Blot",
            "Southern Blot",
            "CRISPR",
            "RNAi",
            "Mutagenesis",
            # Protein Analysis
            "Protein Purification",
            "Mass Spectrometry",
            "Chromatography",
            "Electrophoresis",
            "Protein Crystallization",
            "Surface Plasmon Resonance",
            "BioLayer Interferometry",
            "Protein-Protein Interaction",
            "Enzyme Assay",
            "Kinase Assay",
            # Imaging & Microscopy
            "Confocal Microscopy",
            "Fluorescence Microscopy",
            "Phase Contrast",
            "DIC",
            "Super-Resolution Microscopy",
            "Electron Microscopy",
            "Light Sheet Microscopy",
            "TIRF Microscopy",
            "Multiphoton Microscopy",
            "Brightfield Microscopy",
            # Automation & High-Throughput
            "Liquid Handling",
            "Automated Workstation",
            "High-Throughput Screening",
            "Robotic System",
            "Plate Reader",
            "Automated Imaging",
            "Batch Processing",
        }

        # Research application areas
        self.research_application_tags = {
            # Basic Research Areas
            "Cell Biology",
            "Molecular Biology",
            "Biochemistry",
            "Immunology",
            "Neuroscience",
            "Cancer Research",
            "Stem Cell Research",
            "Developmental Biology",
            "Genetics",
            "Genomics",
            "Proteomics",
            "Metabolomics",
            "Epigenetics",
            "Structural Biology",
            # Disease Areas
            "Oncology",
            "Neurodegenerative Disease",
            "Cardiovascular Disease",
            "Autoimmune Disease",
            "Infectious Disease",
            "Metabolic Disease",
            "Genetic Disorders",
            "Rare Disease",
            # Therapeutic Areas
            "Drug Discovery",
            "Biomarker Discovery",
            "Diagnostics",
            "Companion Diagnostics",
            "Personalized Medicine",
            "Immunotherapy",
            "Gene Therapy",
            "Cell Therapy",
            # Model Systems
            "Human Research",
            "Mouse Model",
            "Rat Model",
            "Cell Line",
            "Primary Cells",
            "Organoid",
            "Tissue Culture",
            "In Vivo",
            "In Vitro",
            "Ex Vivo",
            # Sample Types
            "Blood",
            "Plasma",
            "Serum",
            "Tissue",
            "FFPE",
            "Fresh Frozen",
            "Cell Culture",
            "Bacterial",
            "Viral",
            "Plant",
            "Environmental",
        }

        # Functional/technical characteristics
        self.functional_tags = {
            # Performance Characteristics
            "High Sensitivity",
            "High Specificity",
            "Quantitative",
            "Qualitative",
            "Semi-Quantitative",
            "Real-Time",
            "Endpoint",
            "Kinetic",
            "High-Throughput",
            "Low-Throughput",
            "Multiplexed",
            "Single-Plex",
            "Automated",
            "Manual",
            "Walk-Away",
            # Product Format
            "Kit",
            "Reagent",
            "Instrument",
            "Consumable",
            "Software",
            "Service",
            "Ready-to-Use",
            "Requires Preparation",
            "Single-Use",
            "Reusable",
            # Quality & Compliance
            "Research Use Only",
            "Diagnostic Use",
            "GMP",
            "ISO Certified",
            "FDA Approved",
            "CE Marked",
            "Validated",
            "Pre-Validated",
            "Custom",
            "Standard",
            # Scale & Capacity
            "96-Well",
            "384-Well",
            "1536-Well",
            "Microplate",
            "Tube Format",
            "Slide Format",
            "Benchtop",
            "Portable",
            "Handheld",
            "Large Scale",
            "Small Scale",
            "Pilot Scale",
            # Technology Platform
            "Fluorescence",
            "Chemiluminescence",
            "Colorimetric",
            "Radiometric",
            "Electrochemical",
            "Magnetic",
            "Acoustic",
            "Optical",
            "Digital",
            "Analog",
        }

        # Specimen/Sample tags
        self.specimen_tags = {
            "Human",
            "Mouse",
            "Rat",
            "Rabbit",
            "Bovine",
            "Porcine",
            "Canine",
            "Feline",
            "Non-Human Primate",
            "Sheep",
            "Goat",
            "Horse",
            "Guinea Pig",
            "Hamster",
            "Chicken",
            "Zebrafish",
            "C. elegans",
            "Drosophila",
            "Yeast",
            "E. coli",
            "Plant",
            "Bacterial",
            "Viral",
            "Fungal",
        }

        self.all_tags = (
            self.technique_tags
            | self.research_application_tags
            | self.functional_tags
            | self.specimen_tags
        )

    def get_tags_by_category(self, category_type: str) -> Set[str]:
        """Get tags by type"""
        category_map = {
            "technique": self.technique_tags,
            "research": self.research_application_tags,
            "functional": self.functional_tags,
            "specimen": self.specimen_tags,
            "all": self.all_tags,
        }
        return category_map.get(category_type, set())

    def format_for_prompt(self, max_per_category: int = 20) -> str:
        """Format tags for LLM prompt with limits to manage token usage"""

        # Select most common/important tags for prompt
        technique_subset = list(sorted(self.technique_tags))[:max_per_category]
        research_subset = list(sorted(self.research_application_tags))[
            :max_per_category
        ]
        functional_subset = list(sorted(self.functional_tags))[:max_per_category]

        return f"""
TECHNIQUE TAGS (choose 1-3 most relevant):
{', '.join(technique_subset)}

RESEARCH APPLICATION TAGS (choose 1-2 most relevant):
{', '.join(research_subset)}

FUNCTIONAL TAGS (choose 1-2 most relevant):
{', '.join(functional_subset)}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-YAML CATEGORY SYSTEM WITH TOKEN OPTIMIZATION - FIXED
# ═══════════════════════════════════════════════════════════════════════════════


class MultiYAMLCategorySystem:
    """Manages multiple YAML files with two-stage classification for token optimization"""

    def __init__(self, yaml_directory: str = None):
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.category_files = {}
        self.master_categories = {}
        self.flat_paths = {}
        self.category_keywords = defaultdict(set)

        self.load_yaml_files()
        self.build_master_structure()
        self.build_keyword_index()

    def load_yaml_files(self):
        """Load all YAML category files"""
        # First, check if the specified directory exists
        if not os.path.exists(self.yaml_directory):
            logger.warning(
                f"YAML directory {self.yaml_directory} not found. Checking current directory..."
            )
            self.yaml_directory = "."

        # Look for YAML files in the specified directory
        pattern = os.path.join(self.yaml_directory, "category_structure_*.yaml")
        yaml_files = glob.glob(pattern)

        if not yaml_files:
            # Fallback: look in current directory
            logger.warning(
                "No YAML files found in specified directory. Checking current directory..."
            )
            yaml_files = glob.glob("category_structure_*.yaml")
            if yaml_files:
                self.yaml_directory = "."

        if not yaml_files:
            # Last resort: check parent directory
            parent_pattern = os.path.join("..", "category_structure_*.yaml")
            yaml_files = glob.glob(parent_pattern)
            if yaml_files:
                self.yaml_directory = ".."

        if not yaml_files:
            error_msg = f"""
ERROR: No category_structure_*.yaml files found!

Searched in:
1. {YAML_DIRECTORY}
2. Current directory: {os.getcwd()}
3. Parent directory

Please ensure your YAML files are in one of these locations:
- {YAML_DIRECTORY}/category_structure_*.yaml
- ./category_structure_*.yaml

Available files in current directory:
{os.listdir('.')}
"""
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(
            f"Found {len(yaml_files)} YAML category files in {self.yaml_directory}"
        )

        for yaml_file in yaml_files:
            try:
                # Extract category name from filename
                filename = os.path.basename(yaml_file)
                category_name = filename.replace("category_structure_", "").replace(
                    ".yaml", ""
                )
                category_name = category_name.replace("_", " ").title()

                # Handle special cases
                category_name = category_name.replace(
                    "Bio Imaging", "Bioimaging/Microscopy"
                )
                category_name = category_name.replace(
                    "Rnai Technology", "RNAi Technology"
                )
                category_name = category_name.replace("Pcr", "PCR")

                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.category_files[category_name] = data.get("categories", {})

                logger.info(f"✓ Loaded: {category_name} from {filename}")

            except Exception as e:
                logger.error(f"✗ Error loading {yaml_file}: {e}")

        if not self.category_files:
            raise ValueError("No valid YAML files could be loaded!")

    def build_master_structure(self):
        """Build master category structure for stage 1 classification"""
        self.master_categories = {}

        for category_name, structure in self.category_files.items():
            # Extract top-level categories and some key subcategories for context
            category_info = {}

            for main_cat, subcats in structure.items():
                if isinstance(subcats, dict) and subcats:
                    # Get a few example subcategories for context
                    examples = list(subcats.keys())[:3]
                    category_info[main_cat] = {
                        "subcategory_examples": examples,
                        "total_subcategories": len(subcats),
                    }
                else:
                    category_info[main_cat] = {
                        "subcategory_examples": [],
                        "total_subcategories": 0,
                    }

            self.master_categories[category_name] = category_info

    def build_keyword_index(self):
        """Build keyword index for intelligent category pre-filtering"""
        for category_name, structure in self.category_files.items():
            keywords = set()

            def extract_keywords(node, depth=0):
                if depth > 3:  # Limit recursion
                    return

                if isinstance(node, dict):
                    for key, value in node.items():
                        # Add category names as keywords
                        keywords.update(key.lower().split())
                        extract_keywords(value, depth + 1)
                elif isinstance(node, list):
                    for item in node:
                        if isinstance(item, str):
                            keywords.update(item.lower().split())
                        elif isinstance(item, dict):
                            extract_keywords(item, depth + 1)

            extract_keywords(structure)
            self.category_keywords[category_name] = keywords

    def get_relevant_categories(
        self, product_name: str, description: str = "", max_categories: int = 3
    ) -> List[str]:
        """Pre-filter categories based on keyword matching to reduce token usage"""
        text = f"{product_name} {description}".lower()
        words = set(text.split())

        category_scores = {}
        for category_name, keywords in self.category_keywords.items():
            # Calculate overlap score
            overlap = len(words.intersection(keywords))
            if overlap > 0:
                category_scores[category_name] = overlap

        # Return top matching categories
        sorted_categories = sorted(
            category_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [cat for cat, score in sorted_categories[:max_categories]]

    def get_master_categories_prompt(self) -> str:
        """Get simplified master structure for stage 1 classification"""
        lines = ["AVAILABLE CATEGORY DOMAINS:"]

        for i, (category_name, info) in enumerate(self.master_categories.items(), 1):
            lines.append(f"{i}. {category_name}")
            # Add a few examples for context
            for main_cat, details in list(info.items())[
                :2
            ]:  # Limit to 2 main categories
                examples = details.get("subcategory_examples", [])[:2]  # Limit examples
                if examples:
                    lines.append(f"   └─ {main_cat}: {', '.join(examples)}...")

        return "\n".join(lines)

    def get_detailed_structure(self, category_name: str) -> str:
        """Get detailed structure for specific category (stage 2)"""
        if category_name not in self.category_files:
            return f"Category '{category_name}' not found"

        structure = self.category_files[category_name]

        def format_structure(node, indent=0, max_depth=4):
            if indent >= max_depth:
                return ["  " * indent + "  • [more subcategories...]"]

            result = []
            prefix = "  " * indent

            if isinstance(node, dict):
                for key, value in node.items():
                    result.append(f"{prefix}- {key}")
                    if value and indent < max_depth - 1:
                        result.extend(format_structure(value, indent + 1, max_depth))
            elif isinstance(node, list):
                for i, item in enumerate(node[:8]):  # Limit to 8 items
                    if isinstance(item, str):
                        result.append(f"{prefix}  • {item}")
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            result.append(f"{prefix}  • {sub_key}")
                            if sub_value and indent < max_depth - 1:
                                result.extend(
                                    format_structure(sub_value, indent + 2, max_depth)
                                )
                            break  # Only show first example for brevity
                if len(node) > 8:
                    result.append(f"{prefix}  • ... ({len(node)-8} more items)")

            return result

        lines = [f"DETAILED STRUCTURE FOR: {category_name.upper()}"]
        lines.extend(format_structure(structure))
        return "\n".join(lines)

    def validate_classification_path(
        self, category_name: str, path_components: List[str]
    ) -> Tuple[bool, str]:
        """Validate a classification path within a specific category"""
        if category_name not in self.category_files:
            return False, f"Category '{category_name}' not found"

        structure = self.category_files[category_name]
        current_node = structure
        validated_path = [category_name]

        for component in path_components:
            if not component:
                continue

            if isinstance(current_node, dict):
                if component in current_node:
                    validated_path.append(component)
                    current_node = current_node[component]
                else:
                    # Try fuzzy matching
                    matches = [
                        k for k in current_node.keys() if component.lower() in k.lower()
                    ]
                    if matches:
                        validated_path.append(matches[0])
                        current_node = current_node[matches[0]]
                    else:
                        return (
                            False,
                            f"Invalid path: {' -> '.join(validated_path)} -> '{component}'",
                        )
            elif isinstance(current_node, list):
                # Check if component is in the list
                matches = [
                    item
                    for item in current_node
                    if isinstance(item, str) and component.lower() in item.lower()
                ]
                if matches:
                    validated_path.append(matches[0])
                    break
                else:
                    return (
                        False,
                        f"Invalid path: {' -> '.join(validated_path)} -> '{component}'",
                    )

        return True, " -> ".join(validated_path)


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED TWO-STAGE CLASSIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════


class TwoStageClassifier:
    """Two-stage classification system for token optimization with strict category enforcement"""

    def __init__(
        self, category_system: MultiYAMLCategorySystem, tag_system: EnhancedTagSystem
    ):
        self.category_system = category_system
        self.tag_system = tag_system

    def stage1_classify(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Stage 1: Identify broad category domains with TIGHTENED logic"""

        # Pre-filter relevant categories
        relevant_categories = self.category_system.get_relevant_categories(
            product_name, description
        )

        master_structure = self.category_system.get_master_categories_prompt()

        # TIGHTENED PROMPT - More strict criteria for secondary classification
        prompt = f"""You are a life science product classification expert.

    TASK: Identify 1-2 broad category domains for this product (Stage 1 of 2).

    STRICT RULES FOR SECONDARY CLASSIFICATION:
    - Only assign a secondary domain if BOTH domains are clearly relevant with HIGH confidence
    - Secondary domain must be equally important as primary (not just tangentially related)
    - If unsure about secondary relevance, assign only primary domain
    - Better to have 1 accurate classification than 2 mediocre ones

    CONFIDENCE GUIDELINES:
    - High: Product clearly and definitively fits this domain
    - Medium: Product fits but with some ambiguity
    - Low: Product might fit but uncertain

    {master_structure}

    Product: "{product_name}"
    Description: "{description}"

    Respond with JSON only:
    {{
    "primary_domain": "domain_name",
    "secondary_domain": "domain_name_or_null", 
    "primary_confidence": "High/Medium/Low",
    "secondary_confidence": "High/Medium/Low_or_null",
    "reasoning": "brief explanation of domain selection and why secondary was/wasn't assigned"
    }}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self.strip_code_fences(text)
            result = json.loads(cleaned)

            # Validate domains exist
            valid_domains = list(self.category_system.master_categories.keys())

            primary = result.get("primary_domain")
            secondary = result.get("secondary_domain")
            primary_conf = result.get("primary_confidence", "Low")
            secondary_conf = result.get("secondary_confidence", "Low")

            if primary not in valid_domains:
                primary = self.find_best_match(primary, valid_domains)

            if secondary and secondary not in valid_domains:
                secondary = self.find_best_match(secondary, valid_domains)

            # TIGHTENED LOGIC: Apply strict filtering rules
            secondary = self._apply_strict_secondary_filtering(
                primary, secondary, primary_conf, secondary_conf
            )

            return {
                "primary_domain": primary,
                "secondary_domain": secondary,
                "primary_confidence": primary_conf,
                "secondary_confidence": secondary_conf if secondary else None,
                "reasoning": result.get("reasoning", ""),
                "token_usage": (
                    response.usage.total_tokens if hasattr(response, "usage") else 0
                ),
            }

        except Exception as e:
            logger.error(f"Stage 1 classification error for '{product_name}': {e}")
            return {
                "primary_domain": "Other",
                "secondary_domain": None,
                "primary_confidence": "Low",
                "secondary_confidence": None,
                "reasoning": "Classification failed",
                "token_usage": 0,
            }

    def _apply_strict_secondary_filtering(
        self, primary: str, secondary: str, primary_conf: str, secondary_conf: str
    ) -> Optional[str]:
        """Apply strict filtering rules for secondary domain assignment"""

        # Rule 1: No secondary if primary confidence is Low
        if primary_conf == "Low":
            logger.info(f"Filtered out secondary domain: primary confidence is Low")
            return None

        # Rule 2: Secondary must have High confidence
        if secondary and secondary_conf != "High":
            logger.info(
                f"Filtered out secondary domain '{secondary}': confidence is {secondary_conf}, requires High"
            )
            return None

        # Rule 3: No secondary if same as primary
        if secondary == primary:
            logger.info(f"Filtered out secondary domain: same as primary ({primary})")
            return None

        # Rule 4: Avoid questionable domain combinations
        questionable_combinations = {
            ("Software", "Lab Equipment"),
            ("Software", "Molecular Biology"),
            ("Software", "Cell Biology"),
            # Add more problematic combinations as you identify them
        }

        if secondary and (primary, secondary) in questionable_combinations:
            logger.info(
                f"Filtered out questionable combination: {primary} + {secondary}"
            )
            return None

        if secondary and (secondary, primary) in questionable_combinations:
            logger.info(
                f"Filtered out questionable combination: {secondary} + {primary}"
            )
            return None

        return secondary

    def stage2_classify(
        self, product_name: str, description: str, domain: str
    ) -> Dict[str, Any]:
        """Enhanced Stage 2: Detailed classification with strict category enforcement"""

        # Get available categories at each level to prevent hallucination
        available_categories = self._get_available_categories_hierarchical(domain)

        # Build constrained prompt with explicit options
        prompt = self._build_strict_classification_prompt(
            product_name, description, domain, available_categories
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self.strip_code_fences(text)
            result = json.loads(cleaned)

            # Validate and auto-correct classification path
            validated_result = self._validate_and_auto_correct(domain, result)

            # Add token usage
            validated_result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            return validated_result

        except Exception as e:
            logger.error(
                f"Stage 2 classification error for '{product_name}' in domain '{domain}': {e}"
            )
            return self.get_fallback_classification(domain)

    def _get_available_categories_hierarchical(self, domain: str) -> Dict[str, Any]:
        """Get available categories at each hierarchical level - FIXED VERSION"""
        if domain not in self.category_system.category_files:
            return {}

        structure = self.category_system.category_files[domain]
        hierarchy = {"level1": [], "level2": {}, "level3": {}, "level4": {}}

        def parse_structure(node, current_path, level):
            """Recursively parse the YAML structure"""

            if level == 1:  # Main categories (e.g., "Antibodies")
                for main_cat, main_value in node.items():
                    hierarchy["level1"].append(main_cat)
                    hierarchy["level2"][main_cat] = []

                    if isinstance(main_value, dict) and "subcategories" in main_value:
                        # Parse subcategories level
                        subcats = main_value["subcategories"]
                        if isinstance(subcats, dict):
                            for subcat, subcat_value in subcats.items():
                                hierarchy["level2"][main_cat].append(subcat)

                                level3_key = f"{main_cat}|{subcat}"
                                hierarchy["level3"][level3_key] = []

                                # Parse subsubcategories level
                                if (
                                    isinstance(subcat_value, dict)
                                    and "subsubcategories" in subcat_value
                                ):
                                    subsubcats = subcat_value["subsubcategories"]

                                    if isinstance(subsubcats, dict):
                                        for (
                                            subsubcat,
                                            subsubcat_value,
                                        ) in subsubcats.items():
                                            hierarchy["level3"][level3_key].append(
                                                subsubcat
                                            )

                                            level4_key = (
                                                f"{main_cat}|{subcat}|{subsubcat}"
                                            )
                                            hierarchy["level4"][level4_key] = []

                                            # Parse subsubsubcategories level
                                            if (
                                                isinstance(subsubcat_value, dict)
                                                and "subsubsubcategories"
                                                in subsubcat_value
                                            ):
                                                subsubsubcats = subsubcat_value[
                                                    "subsubsubcategories"
                                                ]
                                                if isinstance(subsubsubcats, list):
                                                    hierarchy["level4"][
                                                        level4_key
                                                    ].extend(
                                                        [
                                                            item
                                                            for item in subsubsubcats
                                                            if isinstance(item, str)
                                                        ]
                                                    )
                                            elif isinstance(subsubcat_value, list):
                                                hierarchy["level4"][level4_key].extend(
                                                    [
                                                        item
                                                        for item in subsubcat_value
                                                        if isinstance(item, str)
                                                    ]
                                                )

                                    elif isinstance(subsubcats, list):
                                        # Direct list of subsubcategories
                                        hierarchy["level3"][level3_key].extend(
                                            [
                                                item
                                                for item in subsubcats
                                                if isinstance(item, str)
                                            ]
                                        )

                                elif isinstance(subcat_value, list):
                                    # Direct list under subcategory
                                    hierarchy["level3"][level3_key].extend(
                                        [
                                            item
                                            for item in subcat_value
                                            if isinstance(item, str)
                                        ]
                                    )

        # Start parsing from the root
        parse_structure(structure, [], 1)

        # Debug output to see what we parsed
        logger.info(f"Parsed hierarchy for {domain}:")
        logger.info(f"  Level 1 categories: {hierarchy['level1'][:3]}...")
        for cat in hierarchy["level1"][:2]:
            level2 = hierarchy["level2"].get(cat, [])
            logger.info(f"  {cat} subcategories: {level2[:3]}...")

        return hierarchy

    def _build_strict_classification_prompt(
        self,
        product_name: str,
        description: str,
        domain: str,
        available_categories: Dict,
    ) -> str:
        """Build prompt with MANDATORY tag assignment - FIXED VERSION"""

        # Get more comprehensive tag lists for the prompt
        technique_tags_list = list(sorted(self.tag_system.technique_tags))[:30]
        research_tags_list = list(sorted(self.tag_system.research_application_tags))[
            :25
        ]
        functional_tags_list = list(sorted(self.tag_system.functional_tags))[:30]

        # Build hierarchical display
        options_display = self._format_available_options_enhanced(
            domain, available_categories
        )

        # Domain-specific tag examples to guide the LLM
        domain_tag_examples = {
            "Lab Equipment": {
                "technique": [
                    "Mass Spectrometry",
                    "Chromatography",
                    "Electrophoresis",
                    "Automated Workstation",
                ],
                "research": ["Cell Biology", "Molecular Biology", "Biochemistry"],
                "functional": [
                    "Instrument",
                    "Automated",
                    "High-Throughput",
                    "Benchtop",
                ],
            },
            "PCR": {
                "technique": ["PCR", "qPCR", "RT-PCR", "DNA Sequencing"],
                "research": ["Molecular Biology", "Genomics", "Diagnostics"],
                "functional": ["Kit", "Reagent", "High-Throughput", "Quantitative"],
            },
            "Immunochemicals": {
                "technique": [
                    "Western Blot",
                    "ELISA",
                    "Immunohistochemistry",
                    "Immunofluorescence",
                ],
                "research": ["Cell Biology", "Cancer Research", "Immunology"],
                "functional": [
                    "Reagent",
                    "Research Use Only",
                    "Diagnostic Use",
                    "High Sensitivity",
                ],
            },
            "RNAi Technology": {
                "technique": ["RNAi", "Gene Expression", "Transfection"],
                "research": ["Molecular Biology", "Cell Biology", "Gene Therapy"],
                "functional": ["Kit", "Reagent", "Research Use Only"],
            },
            "Bioimaging Microscopy": {
                "technique": [
                    "Confocal Microscopy",
                    "Fluorescence Microscopy",
                    "Live Cell Imaging",
                ],
                "research": ["Cell Biology", "Cancer Research", "Neuroscience"],
                "functional": ["Instrument", "Automated", "Digital", "High Resolution"],
            },
            "Antibodies": {
                "technique": [
                    "Western Blot",
                    "Immunofluorescence",
                    "Flow Cytometry",
                    "Immunohistochemistry",
                ],
                "research": ["Cell Biology", "Cancer Research", "Immunology"],
                "functional": [
                    "Research Use Only",
                    "Diagnostic Use",
                    "High Specificity",
                ],
            },
        }

        # Get domain-specific examples
        domain_examples = domain_tag_examples.get(
            domain,
            {
                "technique": ["Cell Culture", "PCR"],
                "research": ["Cell Biology", "Molecular Biology"],
                "functional": ["Research Use Only", "Kit"],
            },
        )

        prompt = f"""You are a life science product classification expert.

    TASK: Classify this product within the {domain} domain and ASSIGN MANDATORY TAGS.

    CRITICAL CLASSIFICATION RULES:
    1. Select categories ONLY from the options provided below
    2. Use EXACT category names - do not modify or create new names
    3. Path format: {domain} -> Main Category -> Subcategory -> Subsubcategory
    4. Must reach at least subsubcategory level (minimum 4 levels including domain)

    {options_display}

    MANDATORY TAG ASSIGNMENT - YOU MUST ASSIGN TAGS:
    You MUST assign 1-3 tags from each category below. Empty tag arrays are NOT ALLOWED.

    TECHNIQUE TAGS (SELECT 1-3 that apply to this product):
    {', '.join(technique_tags_list)}

    RESEARCH APPLICATION TAGS (SELECT 1-2 that apply):
    {', '.join(research_tags_list)}

    FUNCTIONAL TAGS (SELECT 1-3 that apply):
    {', '.join(functional_tags_list)}

    DOMAIN-SPECIFIC TAG GUIDANCE for {domain}:
    - Typical technique tags: {', '.join(domain_examples['technique'])}
    - Typical research tags: {', '.join(domain_examples['research'])}
    - Typical functional tags: {', '.join(domain_examples['functional'])}

    Product: "{product_name}"
    Description: "{description}"

    TAG ASSIGNMENT RULES:
    1. Think about HOW this product is used (technique tags)
    2. Think about WHAT research it supports (research tags)  
    3. Think about WHAT TYPE of product it is (functional tags)
    4. If unsure, select the most general/relevant tags from the domain guidance above
    5. NEVER return empty tag arrays - always select at least 1 tag per category

    EXAMPLE RESPONSES FOR DIFFERENT DOMAINS:

    Lab Equipment Example:
    {{
    "category": "Lab Equipment",
    "subcategory": "Analytical Instrumentation",
    "subsubcategory": "Mass Spectrometers",
    "confidence": "High",
    "technique_tags": ["Mass Spectrometry", "Protein Purification"],
    "research_tags": ["Proteomics", "Biochemistry"],
    "functional_tags": ["Instrument", "High Sensitivity"]
    }}

    PCR Example:
    {{
    "category": "Polymerase Chain Reaction (PCR)",
    "subcategory": "PCR Mixes and Kits", 
    "subsubcategory": "PCR Master Mixes",
    "confidence": "High",
    "technique_tags": ["PCR", "DNA Sequencing"],
    "research_tags": ["Molecular Biology", "Genomics"],
    "functional_tags": ["Kit", "Ready-to-Use"]
    }}

    Respond with JSON only using EXACT category names and MANDATORY non-empty tag arrays:
    {{
    "category": "exact_main_category_name",
    "subcategory": "exact_subcategory_name",
    "subsubcategory": "exact_subsubcategory_name", 
    "subsubsubcategory": "exact_4th_level_if_available",
    "confidence": "High/Medium/Low",
    "technique_tags": ["required_tag1", "required_tag2"],
    "research_tags": ["required_tag1"],
    "functional_tags": ["required_tag1", "required_tag2"]
    }}

    CRITICAL: All three tag arrays must contain at least one tag each. Empty arrays will be rejected."""

        return prompt

    def _format_available_options_enhanced(
        self, domain: str, hierarchy: Dict, max_display: int = 6
    ) -> str:
        """Enhanced formatting of hierarchical options"""
        lines = [f"AVAILABLE CATEGORIES IN {domain.upper()}:"]
        lines.append("=" * 50)

        level1_categories = hierarchy.get("level1", [])

        for i, category in enumerate(level1_categories[:max_display]):
            lines.append(f"\n{i+1}. MAIN CATEGORY: {category}")

            # Show subcategories
            subcategories = hierarchy.get("level2", {}).get(category, [])[:4]
            if subcategories:
                lines.append("   SUBCATEGORIES:")
                for j, subcategory in enumerate(subcategories):
                    lines.append(f"   ├─ {subcategory}")

                    # Show subsubcategories
                    level3_key = f"{category}|{subcategory}"
                    subsubcategories = hierarchy.get("level3", {}).get(level3_key, [])[
                        :3
                    ]
                    if subsubcategories:
                        for k, subsubcategory in enumerate(subsubcategories):
                            symbol = "└─" if k == len(subsubcategories) - 1 else "├─"
                            lines.append(f"   │  {symbol} {subsubcategory}")

                            # Show 4th level if available
                            level4_key = f"{category}|{subcategory}|{subsubcategory}"
                            level4_items = hierarchy.get("level4", {}).get(
                                level4_key, []
                            )[:2]
                            for item in level4_items:
                                lines.append(f"   │     └─ {item}")

                    # Show remaining count
                    remaining_level3 = (
                        len(hierarchy.get("level3", {}).get(level3_key, [])) - 3
                    )
                    if remaining_level3 > 0:
                        lines.append(
                            f"   │  └─ ... ({remaining_level3} more subsubcategories)"
                        )

                # Show remaining subcategories count
                remaining_subcats = (
                    len(subcategories) - 4 if len(subcategories) > 4 else 0
                )
                if remaining_subcats > 0:
                    lines.append(f"   └─ ... ({remaining_subcats} more subcategories)")
            else:
                lines.append("   (No subcategories found)")

        # Show remaining main categories count
        remaining_main = len(level1_categories) - max_display
        if remaining_main > 0:
            lines.append(f"\n... ({remaining_main} more main categories)")

        lines.append(
            "\nIMPORTANT: Use the exact names shown above, NOT structural words like 'subcategories'"
        )

        return "\n".join(lines)

    def _validate_and_auto_correct(self, domain: str, result: Dict) -> Dict[str, Any]:
        """Validate classification and auto-correct to ensure valid paths"""

        # Get hierarchy for validation
        hierarchy = self._get_available_categories_hierarchical(domain)

        # Extract and validate each level
        category = result.get("category", "")
        subcategory = result.get("subcategory", "")
        subsubcategory = result.get("subsubcategory", "")
        subsubsubcategory = result.get("subsubsubcategory", "")

        validated_path = [domain]
        is_valid = True

        # Validate Level 1: Category
        level1_options = hierarchy.get("level1", [])
        if category in level1_options:
            validated_path.append(category)
        elif level1_options:
            # Auto-correct to best match
            category = self._find_best_match_internal(category, level1_options)
            validated_path.append(category)
            logger.info(f"Auto-corrected category to: {category}")
            is_valid = False
        else:
            return self.get_fallback_classification(domain)

        # Validate Level 2: Subcategory
        level2_options = hierarchy.get("level2", {}).get(category, [])
        if subcategory and subcategory in level2_options:
            validated_path.append(subcategory)
        elif level2_options:
            if subcategory:
                subcategory = self._find_best_match_internal(
                    subcategory, level2_options
                )
                logger.info(f"Auto-corrected subcategory to: {subcategory}")
                is_valid = False
            else:
                subcategory = level2_options[0]  # Auto-select first option
            validated_path.append(subcategory)

        # Validate Level 3: Subsubcategory
        level3_key = f"{category}|{subcategory}"
        level3_options = hierarchy.get("level3", {}).get(level3_key, [])
        if subsubcategory and subsubcategory in level3_options:
            validated_path.append(subsubcategory)
        elif level3_options:
            if subsubcategory:
                subsubcategory = self._find_best_match_internal(
                    subsubcategory, level3_options
                )
                logger.info(f"Auto-corrected subsubcategory to: {subsubcategory}")
                is_valid = False
            else:
                subsubcategory = level3_options[0]  # Auto-select first option
            validated_path.append(subsubcategory)

        # Validate Level 4: Subsubsubcategory (optional)
        level4_key = f"{category}|{subcategory}|{subsubcategory}"
        level4_options = hierarchy.get("level4", {}).get(level4_key, [])
        if level4_options and subsubsubcategory:
            if subsubsubcategory in level4_options:
                validated_path.append(subsubsubcategory)
            else:
                subsubsubcategory = self._find_best_match_internal(
                    subsubsubcategory, level4_options
                )
                validated_path.append(subsubsubcategory)
                logger.info(f"Auto-corrected subsubsubcategory to: {subsubsubcategory}")
                is_valid = False
        elif level4_options and len(level4_options) == 1:
            # Auto-select if only one option available
            subsubsubcategory = level4_options[0]
            validated_path.append(subsubsubcategory)

        # Validate tags
        technique_tags = [
            tag
            for tag in result.get("technique_tags", [])
            if tag in self.tag_system.technique_tags
        ]
        research_tags = [
            tag
            for tag in result.get("research_tags", [])
            if tag in self.tag_system.research_application_tags
        ]
        functional_tags = [
            tag
            for tag in result.get("functional_tags", [])
            if tag in self.tag_system.functional_tags
        ]

        return {
            "domain": domain,
            "category": category,
            "subcategory": subcategory,
            "subsubcategory": subsubcategory,
            "subsubsubcategory": subsubsubcategory,
            "confidence": result.get("confidence", "Medium"),
            "technique_tags": technique_tags,
            "research_tags": research_tags,
            "functional_tags": functional_tags,
            "is_valid_path": len(validated_path) >= 4,  # Minimum depth requirement
            "validated_path": " -> ".join(validated_path),
            "was_auto_corrected": not is_valid,
        }

    def _find_best_match_internal(self, target: str, options: List[str]) -> str:
        """Find best matching option from available categories (internal helper)"""
        if not target or not options:
            return options[0] if options else ""

        target_lower = target.lower()

        # Exact match (case-insensitive)
        for option in options:
            if target_lower == option.lower():
                return option

        # Substring match
        for option in options:
            if target_lower in option.lower() or option.lower() in target_lower:
                return option

        # Word overlap match
        target_words = set(target_lower.split())
        best_match = options[0]
        max_overlap = 0

        for option in options:
            option_words = set(option.lower().split())
            overlap = len(target_words.intersection(option_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = option

        return best_match

    def classify_product(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Complete two-stage classification process with tightened logic"""

        # Stage 1: Identify domains with strict filtering
        stage1_result = self.stage1_classify(product_name, description)

        classifications = []
        total_tokens = stage1_result.get("token_usage", 0)

        # Stage 2: Only process domains that passed strict filtering
        domains_to_classify = []

        # Always process primary domain (unless it's "Other" with Low confidence)
        primary_domain = stage1_result["primary_domain"]
        primary_conf = stage1_result["primary_confidence"]

        if primary_domain and primary_domain != "Other":
            domains_to_classify.append(primary_domain)
        elif primary_domain == "Other" and primary_conf != "Low":
            # Only process "Other" if confidence is not Low
            domains_to_classify.append(primary_domain)

        # Only process secondary if it exists and passed filtering
        secondary_domain = stage1_result["secondary_domain"]
        if secondary_domain:
            domains_to_classify.append(secondary_domain)

        # Stage 2: Detailed classification for filtered domains
        for i, domain in enumerate(domains_to_classify):
            stage2_result = self.stage2_classify(product_name, description, domain)

            # Additional quality check: reject Stage 2 results with very low quality
            if self._is_stage2_result_acceptable(stage2_result):
                classification = {
                    "domain": domain,
                    "category": stage2_result.get("category", ""),
                    "subcategory": stage2_result.get("subcategory", ""),
                    "subsubcategory": stage2_result.get("subsubcategory", ""),
                    "subsubsubcategory": stage2_result.get("subsubsubcategory", ""),
                    "confidence": stage2_result.get("confidence", "Low"),
                    "is_primary": i == 0,
                    "is_valid_path": stage2_result.get("is_valid_path", False),
                    "validated_path": stage2_result.get("validated_path", ""),
                    "technique_tags": stage2_result.get("technique_tags", []),
                    "research_tags": stage2_result.get("research_tags", []),
                    "functional_tags": stage2_result.get("functional_tags", []),
                }
                classifications.append(classification)
            else:
                logger.info(
                    f"Rejected Stage 2 result for domain '{domain}' due to low quality"
                )

            total_tokens += stage2_result.get("token_usage", 0)

        # If no classifications passed quality checks, provide minimal fallback
        if not classifications:
            classifications.append(
                {
                    "domain": primary_domain,
                    "category": "Unclassified",
                    "subcategory": "",
                    "subsubcategory": "",
                    "subsubsubcategory": "",
                    "confidence": "Low",
                    "is_primary": True,
                    "is_valid_path": False,
                    "validated_path": f"{primary_domain} -> Unclassified",
                    "technique_tags": [],
                    "research_tags": [],
                    "functional_tags": [],
                }
            )

        # Compile final result
        all_tags = []
        for classification in classifications:
            all_tags.extend(classification.get("technique_tags", []))
            all_tags.extend(classification.get("research_tags", []))
            all_tags.extend(classification.get("functional_tags", []))

        # Remove duplicates while preserving order
        unique_tags = list(dict.fromkeys(all_tags))

        return {
            "classifications": classifications,
            "stage1_domains": {
                "primary": stage1_result["primary_domain"],
                "secondary": stage1_result["secondary_domain"],
            },
            "stage1_confidences": {
                "primary": stage1_result["primary_confidence"],
                "secondary": stage1_result["secondary_confidence"],
            },
            "all_tags": unique_tags,
            "tag_count": len(unique_tags),
            "classification_count": len(classifications),
            "total_token_usage": total_tokens,
            "stage1_reasoning": stage1_result.get("reasoning", ""),
            "filtering_applied": stage1_result["secondary_domain"]
            != stage1_result.get("original_secondary"),  # Track if filtering occurred
        }

    def _is_stage2_result_acceptable(self, stage2_result: Dict[str, Any]) -> bool:
        """Check if Stage 2 result meets minimum quality standards"""

        # Reject if category is "Other" and confidence is Low
        if (
            stage2_result.get("category") == "Other"
            and stage2_result.get("confidence") == "Low"
        ):
            return False

        # Reject if no valid path and confidence is Low
        if (
            not stage2_result.get("is_valid_path", False)
            and stage2_result.get("confidence") == "Low"
        ):
            return False

        # Accept all other cases
        return True

    def find_best_match(self, target: str, options: List[str]) -> str:
        """Find best matching option for invalid domain names"""
        if not target:
            return "Other"

        target_lower = target.lower()

        # Direct substring match
        for option in options:
            if target_lower in option.lower() or option.lower() in target_lower:
                return option

        # Word overlap match
        target_words = set(target_lower.split())
        best_match = "Other"
        max_overlap = 0

        for option in options:
            option_words = set(option.lower().split())
            overlap = len(target_words.intersection(option_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = option

        return best_match if max_overlap > 0 else "Other"

    def strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()

    def get_fallback_classification(self, domain: str) -> Dict[str, Any]:
        """Fallback classification when Stage 2 fails"""
        return {
            "domain": domain,
            "category": "Other",
            "subcategory": "",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "technique_tags": [],
            "research_tags": [],
            "functional_tags": [],
            "is_valid_path": False,
            "validated_path": f"{domain} -> Other",
            "token_usage": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION AND PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_enhanced_validation_sample(
    input_csv: str, sample_size: int = VALIDATION_SAMPLE_SIZE
) -> pd.DataFrame:
    """Create validation sample with enhanced multi-label columns"""
    try:
        df = pd.read_csv(input_csv)

        if len(df) < sample_size:
            logger.warning(
                f"Dataset has only {len(df)} products, using all for validation"
            )
            sample_df = df.copy()
        else:
            sample_indices = random.sample(range(len(df)), sample_size)
            sample_df = df.iloc[sample_indices].copy()

        # Enhanced multi-label classification columns
        sample_df["primary_domain"] = ""
        sample_df["primary_category"] = ""
        sample_df["primary_subcategory"] = ""
        sample_df["primary_subsubcategory"] = ""
        sample_df["primary_subsubsubcategory"] = ""
        sample_df["primary_confidence"] = ""
        sample_df["primary_path_valid"] = False

        sample_df["secondary_domain"] = ""
        sample_df["secondary_category"] = ""
        sample_df["secondary_subcategory"] = ""
        sample_df["secondary_subsubcategory"] = ""
        sample_df["secondary_subsubsubcategory"] = ""
        sample_df["secondary_confidence"] = ""
        sample_df["secondary_path_valid"] = False

        # Tag columns
        sample_df["technique_tags"] = ""
        sample_df["research_tags"] = ""
        sample_df["functional_tags"] = ""
        sample_df["all_tags"] = ""
        sample_df["tag_count"] = 0

        # Metadata
        sample_df["classification_count"] = 0
        sample_df["total_token_usage"] = 0
        sample_df["stage1_reasoning"] = ""
        sample_df["needs_manual_review"] = False

        logger.info(
            f"Created enhanced validation sample with {len(sample_df)} products"
        )
        return sample_df

    except Exception as e:
        logger.error(f"Error creating validation sample: {e}")
        raise


def process_validation_sample():
    """Process validation sample with enhanced two-stage classification"""
    logger.info("Starting enhanced validation sample processing...")

    # Initialize systems
    category_system = MultiYAMLCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = TwoStageClassifier(category_system, tag_system)

    # Create or load validation sample
    if os.path.exists(VALIDATION_CSV):
        response = input(
            f"Validation file {VALIDATION_CSV} exists. Load existing (1) or create new (2)? "
        )
        if response == "1":
            validation_df = pd.read_csv(VALIDATION_CSV)
            logger.info(
                f"Loaded existing validation sample with {len(validation_df)} products"
            )
        else:
            validation_df = create_enhanced_validation_sample(INPUT_CSV)
    else:
        validation_df = create_enhanced_validation_sample(INPUT_CSV)

    # Process unclassified products
    unprocessed_mask = validation_df["primary_domain"].isna() | (
        validation_df["primary_domain"] == ""
    )
    unprocessed_indices = validation_df[unprocessed_mask].index

    if len(unprocessed_indices) > 0:
        logger.info(
            f"Processing {len(unprocessed_indices)} products with enhanced two-stage classification..."
        )

        total_tokens = 0

        for idx in tqdm(unprocessed_indices, desc="Enhanced Classification"):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            # Perform two-stage classification
            result = classifier.classify_product(name, description)

            # Store results
            classifications = result.get("classifications", [])

            # Primary classification
            if classifications:
                primary = classifications[0]
                validation_df.at[idx, "primary_domain"] = primary.get("domain", "")
                validation_df.at[idx, "primary_category"] = primary.get("category", "")
                validation_df.at[idx, "primary_subcategory"] = primary.get(
                    "subcategory", ""
                )
                validation_df.at[idx, "primary_subsubcategory"] = primary.get(
                    "subsubcategory", ""
                )
                validation_df.at[idx, "primary_subsubsubcategory"] = primary.get(
                    "subsubsubcategory", ""
                )
                validation_df.at[idx, "primary_confidence"] = primary.get(
                    "confidence", ""
                )
                validation_df.at[idx, "primary_path_valid"] = primary.get(
                    "is_valid_path", False
                )

            # Secondary classification
            if len(classifications) > 1:
                secondary = classifications[1]
                validation_df.at[idx, "secondary_domain"] = secondary.get("domain", "")
                validation_df.at[idx, "secondary_category"] = secondary.get(
                    "category", ""
                )
                validation_df.at[idx, "secondary_subcategory"] = secondary.get(
                    "subcategory", ""
                )
                validation_df.at[idx, "secondary_subsubcategory"] = secondary.get(
                    "subsubcategory", ""
                )
                validation_df.at[idx, "secondary_subsubsubcategory"] = secondary.get(
                    "subsubsubcategory", ""
                )
                validation_df.at[idx, "secondary_confidence"] = secondary.get(
                    "confidence", ""
                )
                validation_df.at[idx, "secondary_path_valid"] = secondary.get(
                    "is_valid_path", False
                )

            # Tags
            all_tags = result.get("all_tags", [])
            technique_tags = []
            research_tags = []
            functional_tags = []

            for classification in classifications:
                technique_tags.extend(classification.get("technique_tags", []))
                research_tags.extend(classification.get("research_tags", []))
                functional_tags.extend(classification.get("functional_tags", []))

            # Remove duplicates
            technique_tags = list(set(technique_tags))
            research_tags = list(set(research_tags))
            functional_tags = list(set(functional_tags))

            validation_df.at[idx, "technique_tags"] = "|".join(technique_tags)
            validation_df.at[idx, "research_tags"] = "|".join(research_tags)
            validation_df.at[idx, "functional_tags"] = "|".join(functional_tags)
            validation_df.at[idx, "all_tags"] = "|".join(all_tags)
            validation_df.at[idx, "tag_count"] = len(all_tags)

            # Metadata
            validation_df.at[idx, "classification_count"] = result.get(
                "classification_count", 0
            )
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )
            validation_df.at[idx, "stage1_reasoning"] = result.get(
                "stage1_reasoning", ""
            )

            # Flag for manual review if needed
            needs_review = (
                (classifications and classifications[0].get("confidence", "") == "Low")
                or not any(c.get("is_valid_path", False) for c in classifications)
                or len(all_tags) == 0
            )
            validation_df.at[idx, "needs_manual_review"] = needs_review

            total_tokens += result.get("total_token_usage", 0)

    # Save validation results
    validation_df.to_csv(VALIDATION_CSV, index=False)
    logger.info(f"Enhanced validation sample saved to {VALIDATION_CSV}")

    # Generate comprehensive report
    generate_validation_report(validation_df)

    return validation_df


def generate_validation_report(validation_df: pd.DataFrame):
    """Generate comprehensive validation report"""
    print("\n" + "=" * 80)
    print("ENHANCED TWO-STAGE CLASSIFICATION VALIDATION REPORT")
    print("=" * 80)

    total_products = len(validation_df)
    classified_products = len(
        validation_df[
            validation_df["primary_domain"].notna()
            & (validation_df["primary_domain"] != "")
        ]
    )

    print(f"Total products validated: {total_products}")
    print(
        f"Successfully classified: {classified_products} ({classified_products/total_products*100:.1f}%)"
    )

    # Domain distribution
    print(f"\n{'PRIMARY DOMAIN DISTRIBUTION':-^60}")
    domain_counts = validation_df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(10).items():
        print(f"  {domain:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Confidence distribution
    print(f"\n{'CONFIDENCE DISTRIBUTION':-^60}")
    confidence_counts = validation_df["primary_confidence"].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Multi-label statistics
    multi_label_count = len(
        validation_df[
            validation_df["secondary_domain"].notna()
            & (validation_df["secondary_domain"] != "")
        ]
    )
    print(f"\n{'MULTI-LABEL STATISTICS':-^60}")
    print(
        f"  Products with 2 classifications: {multi_label_count} ({multi_label_count/total_products*100:.1f}%)"
    )
    print(f"  Average tags per product: {validation_df['tag_count'].mean():.1f}")
    print(
        f"  Products with tags: {len(validation_df[validation_df['tag_count'] > 0])} ({len(validation_df[validation_df['tag_count'] > 0])/total_products*100:.1f}%)"
    )

    # Tag analysis
    print(f"\n{'TOP TECHNIQUE TAGS':-^60}")
    all_technique_tags = []
    for tags_str in validation_df["technique_tags"].dropna():
        if tags_str:
            all_technique_tags.extend(tags_str.split("|"))

    if all_technique_tags:
        technique_counts = Counter(all_technique_tags)
        for tag, count in technique_counts.most_common(10):
            print(f"  {tag:<35} {count:>5}")

    # Quality metrics
    print(f"\n{'QUALITY METRICS':-^60}")
    valid_paths = len(validation_df[validation_df["primary_path_valid"] == True])
    high_confidence = len(validation_df[validation_df["primary_confidence"] == "High"])
    needs_review = len(validation_df[validation_df["needs_manual_review"] == True])

    print(
        f"  Valid classification paths: {valid_paths} ({valid_paths/total_products*100:.1f}%)"
    )
    print(
        f"  High confidence classifications: {high_confidence} ({high_confidence/total_products*100:.1f}%)"
    )
    print(
        f"  Needs manual review: {needs_review} ({needs_review/total_products*100:.1f}%)"
    )

    # Token usage
    total_tokens = validation_df["total_token_usage"].sum()
    avg_tokens = validation_df["total_token_usage"].mean()
    print(f"\n{'TOKEN USAGE ANALYSIS':-^60}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════
def debug_tagging_test():
    """Quick test to debug tagging issues"""
    print("🔍 DEBUGGING TAGGING ISSUES - PROBLEMATIC PRODUCTS")
    print("=" * 50)

    # Initialize systems
    category_system = MultiYAMLCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = TwoStageClassifier(category_system, tag_system)

    # Test the ACTUAL problematic products from your validation sample
    problematic_products = [
        "ftir spectrophotometer",
        "la taq",
        "horseradish peroxidase hrp",
        "sirna reagent system",
        "megaplex preamp primers",
    ]

    for product_name in problematic_products:
        print(f"\n🧪 Testing: {product_name}")
        result = classifier.classify_product(product_name, "")

        # Show final result
        classifications = result.get("classifications", [])
        if classifications:
            classification = classifications[0]
            print(f"Final tags in classification:")
            print(f"  technique_tags: {classification.get('technique_tags')}")
            print(f"  research_tags: {classification.get('research_tags')}")
            print(f"  functional_tags: {classification.get('functional_tags')}")

        print("\n" + "-" * 30)


def test_classification_examples():
    """Test the system with your specific examples"""
    print("\n" + "=" * 80)
    print("TESTING WITH SPECIFIC EXAMPLES")
    print("=" * 80)

    # Initialize systems
    category_system = MultiYAMLCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = TwoStageClassifier(category_system, tag_system)

    # Test cases
    test_products = [
        {
            "name": "Anti-Bcl-2 antibody [N1N2], N-term",
            "description": "Monoclonal antibody against Bcl-2 protein for cancer research and apoptosis studies",
        },
        {
            "name": "BD FACSCanto II Flow Cytometer",
            "description": "Multi-color flow cytometer for cell analysis and sorting",
        },
        {
            "name": "Human TNF-alpha ELISA Kit",
            "description": "Quantitative ELISA kit for measuring TNF-alpha in human samples",
        },
        {
            "name": "Nikon A1R Confocal Microscope",
            "description": "High-resolution confocal laser scanning microscope",
        },
    ]

    for i, product in enumerate(test_products, 1):
        test_header = f"TEST {i}: {product['name']}"
        print(f"\n{test_header:-^60}")

        result = classifier.classify_product(product["name"], product["description"])

        print(f"Stage 1 Domains: {result['stage1_domains']}")
        print(f"Total Classifications: {result['classification_count']}")
        print(f"Token Usage: {result['total_token_usage']}")

        for j, classification in enumerate(result["classifications"]):
            prefix = "PRIMARY" if classification.get("is_primary") else "SECONDARY"
            print(f"\n{prefix} CLASSIFICATION:")
            print(f"  Domain: {classification['domain']}")
            print(f"  Path: {classification['validated_path']}")
            print(f"  Confidence: {classification['confidence']}")
            print(f"  Valid Path: {classification['is_valid_path']}")

            if classification.get("technique_tags"):
                print(
                    f"  Technique Tags: {', '.join(classification['technique_tags'])}"
                )
            if classification.get("research_tags"):
                print(f"  Research Tags: {', '.join(classification['research_tags'])}")
            if classification.get("functional_tags"):
                print(
                    f"  Functional Tags: {', '.join(classification['functional_tags'])}"
                )


def main():
    """Main execution function"""
    print("=" * 80)
    print("ENHANCED MULTI-YAML TWO-STAGE CLASSIFICATION SYSTEM")
    print("=" * 80)

    print(f"Looking for YAML files in: {YAML_DIRECTORY}")
    print(f"Current working directory: {os.getcwd()}")

    # Initialize and test systems
    try:
        category_system = MultiYAMLCategorySystem()
        tag_system = EnhancedTagSystem()

        print(
            f"✓ Successfully loaded {len(category_system.category_files)} category domains:"
        )
        for domain_name in category_system.category_files.keys():
            print(f"   - {domain_name}")

        print(f"✓ Total available tags: {len(tag_system.all_tags)}")

        # Test with examples
        test_classification_examples()
        debug_tagging_test()

        # Ask user to proceed with validation
        print("\n" + "=" * 80)
        user_input = input("Proceed with validation sample processing? (y/n): ")

        if user_input.lower() == "y":
            validation_df = process_validation_sample()

            print("\n" + "=" * 80)
            print("VALIDATION COMPLETE!")
            print("=" * 80)

    except Exception as e:
        logger.error(f"System initialization error: {e}")
        print(f"ERROR: {e}")
        print("\nTo fix this issue:")
        print("1. Make sure your YAML files are in the correct directory")
        print("2. Check that the directory paths in the configuration are correct")
        print(
            "3. Verify your YAML files have the correct naming pattern: category_structure_*.yaml"
        )


if __name__ == "__main__":
    main()
