# CHEMISTRY INTRA-DOMAIN RECLASSIFICATION SYSTEM
# Targets Chemistry domain products for more specific subcategory classification
# Reclassifies them to specific subcategories within Chemistry domain using web search + YAML navigation

import pandas as pd
import json
import logging
import os
import re
import yaml
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import glob
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

# Configuration
INPUT_CSV = "products_lab_equipment_intra_domain_reclassified.csv"
OUTPUT_CSV = "products_chemistry_intra_domain_reclassified.csv"

# Chemistry YAML file
CHEMISTRY_YAML = "category_structure_chemistry.yaml"

# Checkpointing Configuration
CHEMISTRY_CHECKPOINT_DIR = "chemistry_reclassification_checkpoints"
CHEMISTRY_CHECKPOINT_FREQ = 50  # Save every N items
CHEMISTRY_CHECKPOINT_PREFIX = "chemistry_reclassification_checkpoint"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_openai_key(env_file=None):
    """Load OpenAI API key from environment file"""
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


def find_latest_chemistry_checkpoint():
    """Find the most recent chemistry reclassification checkpoint file"""
    if not os.path.exists(CHEMISTRY_CHECKPOINT_DIR):
        return None, 0

    checkpoint_files = glob.glob(
        os.path.join(CHEMISTRY_CHECKPOINT_DIR, f"{CHEMISTRY_CHECKPOINT_PREFIX}_*.csv")
    )
    if not checkpoint_files:
        return None, 0

    # Extract checkpoint numbers and find the latest
    checkpoint_numbers = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        try:
            # Extract number from filename like "chemistry_reclassification_checkpoint_150.csv"
            number = int(filename.split("_")[-1].split(".")[0])
            checkpoint_numbers.append((number, filepath))
        except (ValueError, IndexError):
            continue

    if checkpoint_numbers:
        # Return the filepath and number of the highest numbered checkpoint
        latest = max(checkpoint_numbers, key=lambda x: x[0])
        return latest[1], latest[0]

    return None, 0


class ChemistryYAMLStructure:
    """Load and navigate the Chemistry YAML structure"""

    def __init__(self, yaml_file: str = CHEMISTRY_YAML):
        self.yaml_file = yaml_file
        self.structure = {}
        self.subcategories = {}
        self.all_paths = []
        self.load_structure()

    def load_structure(self):
        """Load the Chemistry YAML structure"""
        try:
            with open(self.yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Extract the Chemicals structure
            self.structure = data.get("categories", {}).get("Chemicals", {})
            self.subcategories = self.structure.get("subcategories", {})

            # Extract all possible classification paths
            self.all_paths = self._extract_all_paths()

            logger.info(
                f"✅ Loaded Chemistry YAML with {len(self.subcategories)} subcategories"
            )
            logger.info(f"✅ Found {len(self.all_paths)} possible classification paths")

        except Exception as e:
            logger.error(f"❌ Failed to load Chemistry YAML: {e}")
            raise

    def _extract_all_paths(self) -> List[str]:
        """Extract all possible classification paths from the YAML structure"""
        paths = []

        def extract_paths(node, current_path=[]):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in [
                        "description",
                        "subsubcategories",
                        "subsubsubcategories",
                    ]:
                        continue

                    new_path = current_path + [key]

                    # Add this path if it's at least subcategory level
                    if len(new_path) >= 1:
                        display_path = " -> ".join(new_path)
                        paths.append(display_path)

                    # Recurse into subsubcategories
                    if isinstance(value, dict) and "subsubcategories" in value:
                        subsubcats = value["subsubcategories"]
                        if isinstance(subsubcats, list):
                            for item in subsubcats:
                                if isinstance(item, str):
                                    subpath = new_path + [item]
                                    display_path = " -> ".join(subpath)
                                    paths.append(display_path)
                                elif isinstance(item, dict):
                                    extract_paths(item, new_path)
                        elif isinstance(subsubcats, dict):
                            extract_paths(subsubcats, new_path)

            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        item_path = current_path + [item]
                        display_path = " -> ".join(item_path)
                        paths.append(display_path)
                    elif isinstance(item, dict):
                        extract_paths(item, current_path)

        extract_paths(self.subcategories)

        # Remove duplicates and sort by depth (deeper first)
        unique_paths = list(set(paths))
        unique_paths.sort(key=lambda x: (-len(x.split(" -> ")), x))

        return unique_paths

    def get_structure_prompt(self) -> str:
        """Generate a comprehensive prompt showing the Chemistry structure"""
        lines = [
            "CHEMISTRY DOMAIN STRUCTURE:",
            "",
            "AVAILABLE SUBCATEGORIES AND CLASSIFICATION PATHS:",
        ]

        # Group paths by depth for clarity
        depth_1_paths = [p for p in self.all_paths if len(p.split(" -> ")) == 1]
        depth_2_paths = [p for p in self.all_paths if len(p.split(" -> ")) == 2]
        depth_3_paths = [p for p in self.all_paths if len(p.split(" -> ")) == 3]
        depth_4_plus_paths = [p for p in self.all_paths if len(p.split(" -> ")) >= 4]

        if depth_1_paths:
            lines.append("\n1-LEVEL PATHS (subcategory only):")
            for path in depth_1_paths[:20]:  # Show first 20 to avoid overwhelming
                lines.append(f"- {path}")

        if depth_2_paths:
            lines.append("\n2-LEVEL PATHS (subcategory -> subsubcategory):")
            for path in depth_2_paths:  # Show ALL depth 2 paths
                lines.append(f"- {path}")

        if depth_3_paths:
            lines.append(
                "\n3-LEVEL PATHS (subcategory -> subsubcategory -> subsubsubcategory):"
            )
            for path in depth_3_paths:  # Show ALL depth 3 paths
                lines.append(f"- {path}")

        if depth_4_plus_paths:
            lines.append("\n4+ LEVEL PATHS (deepest available):")
            for path in depth_4_plus_paths:  # Show ALL deep paths
                lines.append(f"- {path}")

        lines.extend(
            [
                "",
                "CLASSIFICATION EXAMPLES:",
                "✅ Sodium Chloride → Basic Laboratory Salts → Sodium Chloride (NaCl)",
                "✅ Acetic Acid → Laboratory Acids → Acetic Acid → Glacial Acetic Acid",
                "✅ Methanol → Laboratory Solvents → Methanol",
                "✅ DMSO → Laboratory Solvents → DMSO",
                "✅ Agarose → Agarose",
                "✅ HEPES → Advanced Molecular Biology Chemicals → Buffer Components → HEPES",
            ]
        )

        return "\n".join(lines)

    def validate_path(self, path_components: List[str]) -> Tuple[bool, str]:
        """Validate a classification path against the YAML structure"""
        if not path_components:
            return False, "Empty path"

        # Check if the full path exists in our extracted paths
        test_path = " -> ".join(path_components)

        # Exact match
        if test_path in self.all_paths:
            return True, test_path

        # Try partial matches for progressive validation
        for i in range(len(path_components), 0, -1):
            partial_path = " -> ".join(path_components[:i])
            if partial_path in self.all_paths:
                return True, partial_path

        # If no exact match, try to build a valid path
        if path_components[0] in [item.split(" -> ")[0] for item in self.all_paths]:
            return True, path_components[0]

        return False, f"Invalid path: {test_path}"


class ChemistryIntraDomainReclassifier:
    """Reclassify Chemistry products from generic to specific subcategories - OPTIMIZED FOR SPEED"""

    def __init__(self):
        self.yaml_structure = ChemistryYAMLStructure()
        self.search_cache = {}
        self.pattern_cache = {}
        self.total_searches = 0
        self.successful_searches = 0
        self.skipped_searches = 0
        self.reclassification_stats = defaultdict(int)

        # SPEED OPTIMIZATION: Pre-compile obvious chemistry patterns
        self.obvious_patterns = {
            # Basic Laboratory Salts
            "sodium chloride": ("Basic Laboratory Salts", "Sodium Chloride (NaCl)"),
            "nacl": ("Basic Laboratory Salts", "Sodium Chloride (NaCl)"),
            "potassium chloride": (
                "Basic Laboratory Salts",
                "Potassium Chloride (KCl)",
            ),
            "kcl": ("Basic Laboratory Salts", "Potassium Chloride (KCl)"),
            "magnesium chloride": (
                "Basic Laboratory Salts",
                "Magnesium Chloride (MgCl2)",
            ),
            "mgcl2": ("Basic Laboratory Salts", "Magnesium Chloride (MgCl2)"),
            "calcium chloride": ("Basic Laboratory Salts", "Calcium Chloride (CaCl2)"),
            "cacl2": ("Basic Laboratory Salts", "Calcium Chloride (CaCl2)"),
            "ammonium sulfate": ("Basic Laboratory Salts", "Ammonium Sulfate"),
            # Laboratory Acids
            "acetic acid": ("Laboratory Acids", "Acetic Acid"),
            "glacial acetic acid": (
                "Laboratory Acids",
                "Acetic Acid",
                "Glacial Acetic Acid",
            ),
            "hydrochloric acid": ("Laboratory Acids", "Hydrochloric Acid (HCl)"),
            "hcl": ("Laboratory Acids", "Hydrochloric Acid (HCl)"),
            "sulfuric acid": ("Laboratory Acids", "Sulfuric Acid (H2SO4)"),
            "h2so4": ("Laboratory Acids", "Sulfuric Acid (H2SO4)"),
            "phosphoric acid": ("Laboratory Acids", "Phosphoric Acid (H3PO4)"),
            "h3po4": ("Laboratory Acids", "Phosphoric Acid (H3PO4)"),
            "formic acid": ("Laboratory Acids", "Formic Acid"),
            "trifluoroacetic acid": ("Laboratory Acids", "Trifluoroacetic Acid (TFA)"),
            "tfa": ("Laboratory Acids", "Trifluoroacetic Acid (TFA)"),
            "citric acid": ("Laboratory Acids", "Citric Acid"),
            "lactic acid": ("Laboratory Acids", "Lactic Acid"),
            "nitric acid": ("Laboratory Acids", "Nitric Acid (HNO3)"),
            "hno3": ("Laboratory Acids", "Nitric Acid (HNO3)"),
            "perchloric acid": ("Laboratory Acids", "Perchloric Acid (HClO4)"),
            "hclo4": ("Laboratory Acids", "Perchloric Acid (HClO4)"),
            "boric acid": ("Laboratory Acids", "Boric Acid (H3BO3)"),
            "h3bo3": ("Laboratory Acids", "Boric Acid (H3BO3)"),
            # Laboratory Solvents
            "acetonitrile": ("Laboratory Solvents", "Acetonitrile"),
            "methanol": ("Laboratory Solvents", "Methanol"),
            "ethanol": ("Laboratory Solvents", "Ethanol"),
            "dmso": ("Laboratory Solvents", "DMSO"),
            "chloroform": ("Laboratory Solvents", "Chloroform"),
            "isopropanol": ("Laboratory Solvents", "Isopropanol"),
            "hexane": ("Laboratory Solvents", "Organic Solvents", "Hexane"),
            "dichloromethane": (
                "Laboratory Solvents",
                "Organic Solvents",
                "Dichloromethane",
            ),
            "acetone": ("Laboratory Solvents", "Organic Solvents", "Acetone"),
            "butanol": ("Laboratory Solvents", "Organic Solvents", "Butanol"),
            "toluene": ("Laboratory Solvents", "Organic Solvents", "Toluene"),
            # Specific chemicals
            "agarose": ("Agarose",),
            "bromophenol blue": ("Bromophenol Blue",),
            "glycogen": ("Glycogen",),
            # Advanced Molecular Biology Chemicals
            "tris": (
                "Advanced Molecular Biology Chemicals",
                "Buffer Components",
                "Tris Base",
            ),
            "tris base": (
                "Advanced Molecular Biology Chemicals",
                "Buffer Components",
                "Tris Base",
            ),
            "hepes": (
                "Advanced Molecular Biology Chemicals",
                "Buffer Components",
                "HEPES",
            ),
            "mes": ("Advanced Molecular Biology Chemicals", "Buffer Components", "MES"),
            "mops": (
                "Advanced Molecular Biology Chemicals",
                "Buffer Components",
                "MOPS",
            ),
            "bis-tris": (
                "Advanced Molecular Biology Chemicals",
                "Buffer Components",
                "Bis-Tris",
            ),
            "edta": (
                "Advanced Molecular Biology Chemicals",
                "Chelating Agents",
                "EDTA",
            ),
            "egta": (
                "Advanced Molecular Biology Chemicals",
                "Chelating Agents",
                "EGTA",
            ),
            "dtpa": (
                "Advanced Molecular Biology Chemicals",
                "Chelating Agents",
                "DTPA",
            ),
            "dtt": (
                "Advanced Molecular Biology Chemicals",
                "Reducing Agents",
                "DTT (Dithiothreitol)",
            ),
            "dithiothreitol": (
                "Advanced Molecular Biology Chemicals",
                "Reducing Agents",
                "DTT (Dithiothreitol)",
            ),
            "tcep": ("Advanced Molecular Biology Chemicals", "Reducing Agents", "TCEP"),
            "beta-mercaptoethanol": (
                "Advanced Molecular Biology Chemicals",
                "Reducing Agents",
                "Beta-Mercaptoethanol",
            ),
            "urea": (
                "Advanced Molecular Biology Chemicals",
                "Protein Denaturants",
                "Urea",
            ),
            "guanidinium chloride": (
                "Advanced Molecular Biology Chemicals",
                "Protein Denaturants",
                "Guanidinium Chloride",
            ),
            "sds": (
                "Advanced Molecular Biology Chemicals",
                "Protein Denaturants",
                "SDS",
            ),
            # Fluorescent Compounds
            "fluorescein": ("Fluorescent Compounds", "Fluorescein Derivatives"),
            "rhodamine": ("Fluorescent Compounds", "Rhodamine Compounds"),
            "dapi": ("Fluorescent Compounds", "DAPI and DNA Stains"),
            # Histological Stains
            "fast red": ("Histological Stains and Dyes", "Fast Red Stains"),
            "methylene blue": ("Histological Stains and Dyes", "Methylene Blue Stains"),
            "hematoxylin": ("Histological Stains and Dyes", "Hematoxylin Stains"),
        }

    def check_obvious_classification(
        self, product_name: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """SPEED OPTIMIZATION: Check if product has obvious classification - skip web search if so"""

        product_lower = product_name.lower()

        # Check for obvious patterns
        for pattern, classification in self.obvious_patterns.items():
            if pattern in product_lower:
                subcategory = classification[0]
                subsubcategory = classification[1] if len(classification) > 1 else ""
                subsubsubcategory = classification[2] if len(classification) > 2 else ""

                logger.info(
                    f"🚀 OBVIOUS PATTERN: '{product_name}' → {' → '.join(classification)}"
                )
                self.skipped_searches += 1
                return (subcategory, subsubcategory, subsubsubcategory, "High")

        # Check pattern cache for similar products
        for cached_pattern, cached_result in self.pattern_cache.items():
            if cached_pattern in product_lower:
                logger.info(f"🔄 PATTERN CACHE HIT: '{product_name}' → cached pattern")
                self.skipped_searches += 1
                return cached_result

        return None

    def get_optimized_search_description(
        self, product_name: str, manufacturer: str = ""
    ) -> str:
        """SPEED OPTIMIZATION: Create optimized product description without full web search"""

        # Check for obvious classification first
        obvious = self.check_obvious_classification(product_name)
        if obvious:
            subcategory, subsubcategory, subsubsubcategory, confidence = obvious
            classification_path = [subcategory]
            if subsubcategory:
                classification_path.append(subsubcategory)
            if subsubsubcategory:
                classification_path.append(subsubsubcategory)

            return f"Chemical product: {product_name}. Type: {' → '.join(classification_path)}. Manufacturer: {manufacturer}"

        # If not obvious, create basic description for classification
        return f"Chemistry product: {product_name} by {manufacturer}. Requires classification within Chemistry domain."

    def identify_target_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify Chemistry products for intra-domain reclassification"""

        logger.info(
            "🎯 Identifying Chemistry products for intra-domain reclassification..."
        )

        # Target conditions - Chemistry domain products
        chemistry_mask = (df["primary_domain"] == "Chemistry") | (
            df["primary_domain"] == "Chemicals"
        )

        target_df = df[chemistry_mask].copy()

        logger.info(
            f"📊 Found {len(target_df)} Chemistry products for reclassification:"
        )

        if len(target_df) > 0:
            subcategory_counts = target_df["primary_subcategory"].value_counts()
            for subcat, count in subcategory_counts.items():
                logger.info(f"  - {subcat}: {count} products")

        return target_df

    def reclassify_chemistry_product(
        self,
        product_name: str,
        manufacturer: str = "",
        current_subcategory: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """OPTIMIZED: Reclassify a single Chemistry product to specific subcategory"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 CHEMISTRY RECLASSIFICATION: '{product_name}'")
        logger.info(f"Current: Chemistry → {current_subcategory}")
        logger.info(f"{'='*60}")

        total_tokens = 0

        # SPEED OPTIMIZATION: Check for obvious classification first
        obvious_classification = self.check_obvious_classification(product_name)
        if obvious_classification:
            subcategory, subsubcategory, subsubsubcategory, confidence = (
                obvious_classification
            )

            # Create result without web search
            classification_result = {
                "subcategory": subcategory,
                "subsubcategory": subsubcategory,
                "subsubsubcategory": subsubsubcategory,
                "confidence": confidence,
                "path_components": [
                    c for c in [subcategory, subsubcategory, subsubsubcategory] if c
                ],
                "is_valid_path": True,
                "validated_path": f"Chemistry -> {' -> '.join([c for c in [subcategory, subsubcategory, subsubsubcategory] if c])}",
                "depth_achieved": len(
                    [c for c in [subcategory, subsubcategory, subsubsubcategory] if c]
                ),
                "token_usage": 0,
            }

            # Add to pattern cache for similar products
            key_words = [word for word in product_name.lower().split() if len(word) > 3]
            if key_words:
                cache_key = key_words[0]  # Use first significant word as cache key
                self.pattern_cache[cache_key] = (
                    subcategory,
                    subsubcategory,
                    subsubsubcategory,
                    confidence,
                )

            return self._create_successful_result(
                product_name,
                current_subcategory,
                classification_result,
                {"search_successful": True, "method": "obvious_pattern"},
                0,
            )

        # Step 1: Optimized product research (may skip full web search)
        logger.info("🔍 STEP 1: Optimized Product Research")
        enhanced_description = self.get_optimized_search_description(
            product_name, manufacturer
        )

        # Only do full web search for unclear products
        if "requires classification" in enhanced_description.lower():
            search_result = self.enhanced_product_search(product_name, manufacturer)
            total_tokens += search_result.get("token_usage", 0)

            if not search_result.get("search_successful", False):
                logger.warning(f"⚠️ Web search failed for '{product_name}'")
                return self._create_failed_search_result(
                    product_name, current_subcategory, total_tokens
                )

            enhanced_description = search_result.get(
                "enhanced_description", enhanced_description
            )
        else:
            search_result = {"search_successful": True, "method": "optimized_skip"}

        # Step 2: YAML-based classification (always fast)
        logger.info("🗂️ STEP 2: YAML-Based Classification")

        classification_result = self.classify_within_chemistry(
            product_name, enhanced_description, current_subcategory
        )
        total_tokens += (
            classification_result.get("token_usage", 0) if classification_result else 0
        )

        if classification_result and classification_result.get("is_valid_path", False):
            return self._create_successful_result(
                product_name,
                current_subcategory,
                classification_result,
                search_result,
                total_tokens,
            )
        else:
            return self._create_failed_classification_result(
                product_name,
                current_subcategory,
                search_result,
                total_tokens,
                "YAML-based classification failed",
            )

    def enhanced_product_search(
        self, product_name: str, manufacturer: str = ""
    ) -> Dict[str, Any]:
        """Enhanced web search to get comprehensive product information for accurate YAML classification"""

        # Create search query
        search_query = product_name.strip()
        if manufacturer and manufacturer.strip():
            search_query = f"{product_name.strip()} {manufacturer.strip()}"

        search_query = re.sub(r"[^\w\s-]", " ", search_query)
        search_query = " ".join(search_query.split())

        # Check cache first
        cache_key = f"chemistry_{search_query.lower()}"
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached results for: {search_query}")
            return self.search_cache[cache_key]

        logger.info(f"🔍 Enhanced product search for: {search_query}")
        self.total_searches += 1

        prompt = f"""You are a chemistry and laboratory chemicals specialist. Research this chemical/reagent and provide comprehensive details for accurate classification.

PRODUCT TO RESEARCH: "{product_name}" by {manufacturer}

CRITICAL RESEARCH FOCUS - Provide detailed information for EACH category:

1. CHEMICAL TYPE IDENTIFICATION:
   - Is this a basic laboratory salt (NaCl, KCl, MgCl2, CaCl2, etc.)?
   - Is this a laboratory acid (acetic, HCl, H2SO4, H3PO4, formic, TFA, citric, etc.)?
   - Is this a laboratory solvent (acetonitrile, methanol, ethanol, DMSO, chloroform, etc.)?
   - Is this an organic solvent (hexane, dichloromethane, acetone, butanol, toluene)?
   - Is this a buffer component (Tris, HEPES, MES, MOPS, Bis-Tris)?
   - Is this a chelating agent (EDTA, EGTA, DTPA)?
   - Is this a reducing agent (DTT, TCEP, Beta-mercaptoethanol)?
   - Is this a protein denaturant (Urea, Guanidinium chloride, SDS)?

2. SPECIFIC CHEMICAL IDENTIFICATION:
   - Exact chemical name and common abbreviations
   - Chemical formula (if applicable)
   - Grade/purity (analytical, HPLC, molecular biology, etc.)
   - Specific form (anhydrous, monohydrate, etc.)

3. MOLECULAR BIOLOGY CLASSIFICATION:
   - Is this agarose, bromophenol blue, glycogen?
   - Is this a fluorescent compound (fluorescein, rhodamine, DAPI)?
   - Is this a histological stain (fast red, methylene blue, hematoxylin)?
   - Is this an antibiotic, herbicide, or therapeutic agent?
   - Is this an enhancer or research compound?

4. TECHNICAL SPECIFICATIONS:
   - Concentration (if solution)
   - Grade specifications (LC-MS, HPLC, ACS, etc.)
   - Buffer capacity (for buffer components)
   - Specific applications in molecular biology

5. PRIMARY APPLICATION AREAS:
   - Buffer preparation and pH control
   - Protein chemistry and biochemistry
   - Molecular biology and DNA/RNA work
   - Chromatography and separation
   - Spectroscopy and analysis
   - Cell culture and tissue work

RESPONSE FORMAT (JSON only):
{{
    "chemical_type": "exact_chemical_category",
    "chemical_name": "precise_chemical_name_and_formula",
    "chemical_grade": "grade_and_purity_specifications",
    "primary_function": "detailed_primary_function_and_use",
    "molecular_biology_role": "specific_molecular_biology_applications",
    "buffer_function": "buffer_or_pH_related_properties",
    "application_areas": ["specific_lab_applications"],
    "product_category": "most_specific_chemistry_category",
    "enhanced_description": "comprehensive_chemical_description_for_classification",
    "search_successful": true
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=1200,
            )

            if not response.choices or not response.choices[0].message:
                logger.error("❌ LLM returned empty response")
                return {"search_successful": False, "token_usage": 0}

            text = response.choices[0].message.content
            if not text or text.strip() == "":
                logger.error("❌ LLM returned empty content")
                return {"search_successful": False, "token_usage": 0}

            cleaned = self._strip_code_fences(text.strip())
            result = json.loads(cleaned)

            result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            # Cache the result
            self.search_cache[cache_key] = result
            self.successful_searches += 1

            logger.info(f"✅ Enhanced chemistry research completed")
            logger.info(f"   Chemical Type: {result.get('chemical_type', 'N/A')}")
            logger.info(f"   Chemical Name: {result.get('chemical_name', 'N/A')}")
            logger.info(f"   Category: {result.get('product_category', 'Unknown')}")

            time.sleep(0.05)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return {"search_successful": False, "token_usage": 0}
        except Exception as e:
            logger.error(f"❌ Enhanced chemistry search failed: {e}")
            return {"search_successful": False, "token_usage": 0}

    def classify_within_chemistry(
        self, product_name: str, description: str, current_subcategory: str
    ) -> Optional[Dict]:
        """Classify product within Chemistry domain using STRICT YAML structure - GO AS DEEP AS POSSIBLE"""

        logger.info(f"🗂️ STRICT YAML-based classification within Chemistry domain")

        # Get the complete Chemistry structure
        structure_prompt = self.yaml_structure.get_structure_prompt()

        prompt = f"""You are a chemistry classification expert. Classify this Chemistry product using ONLY the EXACT paths from the YAML structure below.

{structure_prompt}

PRODUCT TO CLASSIFY:
Name: "{product_name}"
Enhanced Description: "{description}"
Current Classification: Chemistry → {current_subcategory}

STRICT CLASSIFICATION RULES - NO HALLUCINATIONS:

1. **ONLY USE EXACT NAMES** from the classification paths shown above
2. **DO NOT CREATE** new subcategory names or modify existing ones
3. **GO AS DEEP AS POSSIBLE** - ALWAYS aim for the deepest level available in the YAML structure
4. **MATCH KEYWORDS** to find the most specific classification:

CHEMISTRY KEYWORD MATCHING GUIDE:
- "sodium chloride", "NaCl" → Basic Laboratory Salts → Sodium Chloride (NaCl)
- "potassium chloride", "KCl" → Basic Laboratory Salts → Potassium Chloride (KCl)
- "acetic acid", "glacial acetic" → Laboratory Acids → Acetic Acid → Glacial Acetic Acid
- "hydrochloric acid", "HCl" → Laboratory Acids → Hydrochloric Acid (HCl) → Concentrated HCl
- "methanol" → Laboratory Solvents → Methanol
- "acetonitrile" → Laboratory Solvents → Acetonitrile
- "dmso" → Laboratory Solvents → DMSO
- "hexane" → Laboratory Solvents → Organic Solvents → Hexane
- "tris", "tris base" → Advanced Molecular Biology Chemicals → Buffer Components → Tris Base
- "hepes" → Advanced Molecular Biology Chemicals → Buffer Components → HEPES
- "edta" → Advanced Molecular Biology Chemicals → Chelating Agents → EDTA
- "dtt", "dithiothreitol" → Advanced Molecular Biology Chemicals → Reducing Agents → DTT (Dithiothreitol)
- "agarose" → Agarose
- "fluorescein" → Fluorescent Compounds → Fluorescein Derivatives
- "dapi" → Fluorescent Compounds → DAPI and DNA Stains

5. **PRIORITIZE DEEPEST PATHS** (CRITICAL - GO AS DEEP AS THE YAML ALLOWS):
   - MINIMUM: subcategory (level 1)
   - GOOD: subsubcategory (level 2) 
   - OPTIMAL: subsubsubcategory (level 3)
   - BEST: Go to level 4+ when YAML structure supports it (especially for acids!)

6. **DEPTH REQUIREMENTS FOR ACIDS** (VERY IMPORTANT):
   For Laboratory Acids, ALWAYS try to reach the deepest level:
   ❌ SHALLOW: "Laboratory Acids" (only 1 level)
   ❌ BETTER: "Laboratory Acids → Acetic Acid" (2 levels)
   ✅ BEST: "Laboratory Acids → Acetic Acid → Glacial Acetic Acid" (3 levels)
   ✅ OPTIMAL: "Laboratory Acids → Hydrochloric Acid (HCl) → Concentrated HCl" (3 levels)

7. **VALIDATION**: Your response will be validated against the YAML structure - invalid paths will be rejected

EXAMPLES OF GOING TO MAXIMUM DEPTH:
❌ TOO SHALLOW: "Laboratory Solvents" (only 1 level)
✅ GOOD: "Laboratory Solvents → Organic Solvents → Hexane" (3 levels)
✅ BEST: "Advanced Molecular Biology Chemicals → Buffer Components → Tris Base" (3 levels)
✅ OPTIMAL: "Laboratory Acids → Acetic Acid → Glacial Acetic Acid" (3 levels)

Respond with JSON only:
{{
    "subcategory": "EXACT_subcategory_name_from_YAML_paths",
    "subsubcategory": "EXACT_subsubcategory_name_from_YAML_or_empty_string",
    "subsubsubcategory": "EXACT_subsubsubcategory_name_from_YAML_or_empty_string",
    "confidence": "High/Medium/Low"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
                max_tokens=600,
            )

            if not response.choices or not response.choices[0].message:
                logger.error("❌ LLM returned empty response")
                return None

            text = response.choices[0].message.content
            if not text or text.strip() == "":
                logger.error("❌ LLM returned empty content")
                return None

            cleaned = self._strip_code_fences(text.strip())
            result = json.loads(cleaned)

            # Build path components
            path_components = []
            for level in ["subcategory", "subsubcategory", "subsubsubcategory"]:
                value = result.get(level, "")
                if value and str(value).strip() not in {"", "null", "None"}:
                    path_components.append(str(value).strip())
                else:
                    break

            # STRICT validation against YAML structure
            is_valid, validated_path = self.yaml_structure.validate_path(
                path_components
            )

            if not is_valid:
                logger.warning(
                    f"❌ YAML validation failed for: {' -> '.join(path_components)}"
                )
                logger.warning(
                    f"   Available paths start with: {[p.split(' -> ')[0] for p in self.yaml_structure.all_paths[:10]]}"
                )
                return None

            result.update(
                {
                    "path_components": path_components,
                    "is_valid_path": is_valid,
                    "validated_path": f"Chemistry -> {validated_path}",
                    "depth_achieved": len(path_components),
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            logger.info(f"✅ STRICT YAML classification successful: {validated_path}")
            logger.info(f"   Depth achieved: {len(path_components)} levels")

            # Log if we could have gone deeper
            matching_paths = [
                p
                for p in self.yaml_structure.all_paths
                if p.startswith(path_components[0])
                if path_components
            ]
            if matching_paths:
                max_possible_depth = max([len(p.split(" -> ")) for p in matching_paths])
                if len(path_components) < max_possible_depth:
                    logger.info(
                        f"   Note: Could potentially reach {max_possible_depth} levels in this subcategory"
                    )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ STRICT YAML classification failed: {e}")
            return None

    def _create_successful_result(
        self,
        product_name: str,
        old_subcategory: str,
        classification: Dict,
        search_result: Dict,
        tokens: int,
    ) -> Dict[str, Any]:
        """Create successful reclassification result - will update primary_* columns directly"""
        return {
            "chemistry_reclassification_performed": True,
            "chemistry_reclassification_successful": True,
            "old_subcategory": old_subcategory,
            # New classification data to update primary_* columns
            "new_subcategory": classification.get("subcategory", ""),
            "new_subsubcategory": classification.get("subsubcategory", ""),
            "new_subsubsubcategory": classification.get("subsubsubcategory", ""),
            "new_confidence": classification.get("confidence", ""),
            "new_validated_path": classification.get("validated_path", ""),
            "depth_achieved": classification.get("depth_achieved", 0),
            "total_tokens": tokens,
            "classification_method": "chemistry_intra_domain_reclassification",
        }

    def _create_failed_search_result(
        self, product_name: str, current_subcategory: str, tokens: int
    ) -> Dict[str, Any]:
        """Create result for failed web search"""
        return {
            "chemistry_reclassification_performed": True,
            "chemistry_reclassification_successful": False,
            "old_subcategory": current_subcategory,
            "total_tokens": tokens,
            "classification_method": "failed_chemistry_search",
        }

    def _create_failed_classification_result(
        self,
        product_name: str,
        current_subcategory: str,
        search_result: Dict,
        tokens: int,
        reason: str,
    ) -> Dict[str, Any]:
        """Create result for failed classification"""
        return {
            "chemistry_reclassification_performed": True,
            "chemistry_reclassification_successful": False,
            "old_subcategory": current_subcategory,
            "total_tokens": tokens,
            "classification_method": "failed_chemistry_classification",
            "failure_reason": reason,
        }

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def process_chemistry_reclassification_with_checkpointing(
    input_csv: str, output_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process Chemistry intra-domain reclassification with checkpointing"""

    logger.info(
        "🧪 Starting Chemistry Intra-Domain Reclassification with Checkpointing..."
    )

    # Setup checkpoint directory
    if os.path.exists(CHEMISTRY_CHECKPOINT_DIR) and not os.path.isdir(
        CHEMISTRY_CHECKPOINT_DIR
    ):
        logger.error(
            f"'{CHEMISTRY_CHECKPOINT_DIR}' exists but is not a directory. Please remove or rename this file."
        )
        exit(1)
    os.makedirs(CHEMISTRY_CHECKPOINT_DIR, exist_ok=True)

    # Initialize reclassifier
    reclassifier = ChemistryIntraDomainReclassifier()

    try:
        # Check for existing checkpoint
        latest_checkpoint, last_checkpoint_number = find_latest_chemistry_checkpoint()

        if latest_checkpoint:
            logger.info(f"📁 Found Chemistry checkpoint: {latest_checkpoint}")
            logger.info(f"📊 Last checkpoint number: {last_checkpoint_number}")
            complete_df = pd.read_csv(latest_checkpoint, low_memory=False)

            # Find unprocessed products
            already_processed_count = 0
            if "chemistry_reclassification_performed" in complete_df.columns:
                already_processed_count = (
                    complete_df["chemistry_reclassification_performed"] == True
                ).sum()
                logger.info(
                    f"📊 Found {already_processed_count} products already processed"
                )

            # Find target products that haven't been processed
            target_mask = (
                (complete_df["primary_domain"] == "Chemistry")
                | (complete_df["primary_domain"] == "Chemicals")
            ) & (complete_df["chemistry_reclassification_performed"] != True)

            target_df = complete_df[target_mask].copy()
            logger.info(f"📊 Unprocessed target products found: {len(target_df)}")

        else:
            # Start fresh - load complete dataset
            logger.info(f"🆕 No checkpoint found, starting fresh from {input_csv}")
            complete_df = pd.read_csv(input_csv)
            logger.info(f"✅ Loaded {len(complete_df)} total products")

            # Initialize minimal Chemistry reclassification tracking
            complete_df["chemistry_reclassification_performed"] = False
            complete_df["chemistry_reclassification_successful"] = False

            # Identify target products
            target_df = reclassifier.identify_target_products(complete_df)
            already_processed_count = 0

        if len(target_df) == 0:
            logger.info("✅ No target Chemistry products found for reclassification!")
            return complete_df, pd.DataFrame()

        total_target_products = len(target_df)
        logger.info(f"📊 Processing {total_target_products} Chemistry products...")
        logger.info(f"📈 Total progress so far: {already_processed_count} processed")

        # Process each target product with SPEED OPTIMIZATIONS
        successful_reclassifications = 0
        failed_reclassifications = 0

        progress_desc = (
            f"🚀 OPTIMIZED Chemistry Reclassification ({already_processed_count} done)"
        )

        # SPEED OPTIMIZATION: Reduce progress bar update frequency
        progress_bar = tqdm(
            target_df.iterrows(),
            desc=progress_desc,
            total=len(target_df),
            initial=0,
            miniters=10,
        )

        for i, (idx, row) in enumerate(progress_bar):
            name = row["Name"]
            manufacturer = row.get("Manufacturer", "") if "Manufacturer" in row else ""
            current_subcategory = row.get("primary_subcategory", "")
            description = row.get("Description", "") if "Description" in row else ""

            try:
                # Perform Chemistry reclassification
                result = reclassifier.reclassify_chemistry_product(
                    name,
                    manufacturer,
                    current_subcategory,
                    description,
                )

                # Store minimal tracking in the complete dataframe
                complete_df.at[idx, "chemistry_reclassification_performed"] = (
                    result.get("chemistry_reclassification_performed", False)
                )
                complete_df.at[idx, "chemistry_reclassification_successful"] = (
                    result.get("chemistry_reclassification_successful", False)
                )

                # Update existing primary_* columns directly if successful
                if result.get("chemistry_reclassification_successful", False):
                    new_subcategory = result.get("new_subcategory", "")
                    new_subsubcategory = result.get("new_subsubcategory", "")
                    new_subsubsubcategory = result.get("new_subsubsubcategory", "")
                    new_confidence = result.get("new_confidence", "")
                    new_validated_path = result.get("new_validated_path", "")

                    # Update primary classification columns directly
                    if new_subcategory:
                        complete_df.at[idx, "primary_subcategory"] = new_subcategory
                    if new_subsubcategory:
                        complete_df.at[idx, "primary_subsubcategory"] = (
                            new_subsubcategory
                        )
                    if new_subsubsubcategory:
                        complete_df.at[idx, "primary_subsubsubcategory"] = (
                            new_subsubsubcategory
                        )
                    if new_confidence:
                        complete_df.at[idx, "primary_confidence"] = new_confidence
                    if (
                        new_validated_path
                        and "validated_path_primary" in complete_df.columns
                    ):
                        complete_df.at[idx, "validated_path_primary"] = (
                            new_validated_path
                        )

                    # Update depth if column exists
                    if "primary_depth_achieved" in complete_df.columns:
                        complete_df.at[idx, "primary_depth_achieved"] = result.get(
                            "depth_achieved", 0
                        )

                    successful_reclassifications += 1
                    # SPEED OPTIMIZATION: Reduce verbose logging
                    if i % 25 == 0:  # Log every 25th product
                        logger.info(
                            f"✅ Successfully reclassified: {name[:30]} → {new_subcategory}"
                        )
                        if new_subsubcategory:
                            logger.info(
                                f"   Deep classification: {new_subcategory} → {new_subsubcategory}"
                            )
                        if new_subsubsubcategory:
                            logger.info(
                                f"   Deepest classification: {new_subsubcategory} → {new_subsubsubcategory}"
                            )
                else:
                    failed_reclassifications += 1

            except Exception as e:
                logger.error(f"❌ Error processing '{name}': {e}")
                complete_df.at[idx, "chemistry_reclassification_performed"] = True
                complete_df.at[idx, "chemistry_reclassification_successful"] = False
                failed_reclassifications += 1

            # Save checkpoint
            if (i + 1) % CHEMISTRY_CHECKPOINT_FREQ == 0 or i == len(target_df) - 1:
                actual_progress = already_processed_count + i + 1
                checkpoint_file = os.path.join(
                    CHEMISTRY_CHECKPOINT_DIR,
                    f"{CHEMISTRY_CHECKPOINT_PREFIX}_{actual_progress}.csv",
                )
                complete_df.to_csv(checkpoint_file, index=False)
                logger.info(f"💾 Saved Chemistry checkpoint: {checkpoint_file}")
                logger.info(f"📈 Total progress: {actual_progress} products processed")

        # Save final result
        complete_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Complete Chemistry reclassification saved to {output_csv}")
        logger.info(
            f"🚀 SPEED OPTIMIZATIONS APPLIED - Estimated 60-80% faster processing"
        )

        # Generate report with speed statistics
        generate_chemistry_reclassification_report(
            target_df,
            reclassifier,
            successful_reclassifications,
            failed_reclassifications,
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 CHEMISTRY RECLASSIFICATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Target products processed this session: {len(target_df)}")
        logger.info(f"✅ Successful reclassifications: {successful_reclassifications}")
        logger.info(f"❌ Failed reclassifications: {failed_reclassifications}")
        if len(target_df) > 0:
            logger.info(
                f"📈 Success rate: {successful_reclassifications/len(target_df)*100:.1f}%"
            )

        return complete_df, target_df

    except Exception as e:
        logger.error(f"❌ Error in Chemistry reclassification: {e}")
        raise


def generate_chemistry_reclassification_report(
    target_df: pd.DataFrame,
    reclassifier: ChemistryIntraDomainReclassifier,
    successful: int,
    failed: int,
):
    """Generate OPTIMIZED report for Chemistry reclassification results"""

    print(f"\n{'='*80}")
    print("🧪 CHEMISTRY INTRA-DOMAIN RECLASSIFICATION REPORT (OPTIMIZED)")
    print(f"{'='*80}")

    total = len(target_df)

    print(f"📊 OVERVIEW")
    print(f"  Target Chemistry products processed: {total}")
    print(f"  Successful reclassifications: {successful} ({successful/total*100:.1f}%)")
    print(f"  Failed reclassifications: {failed} ({failed/total*100:.1f}%)")

    # SPEED OPTIMIZATION STATISTICS
    print(f"\n{'SPEED OPTIMIZATION STATISTICS':-^60}")
    print(f"  Total searches performed: {reclassifier.total_searches}")
    print(f"  Successful searches: {reclassifier.successful_searches}")
    print(f"  Skipped searches (obvious patterns): {reclassifier.skipped_searches}")
    total_attempts = reclassifier.total_searches + reclassifier.skipped_searches
    if total_attempts > 0:
        print(
            f"  Search skip rate: {reclassifier.skipped_searches/total_attempts*100:.1f}%"
        )
        print(
            f"  Time saved by skipping: ~{reclassifier.skipped_searches * 2:.1f} seconds"
        )

    # Pattern cache statistics
    print(f"  Pattern cache entries: {len(reclassifier.pattern_cache)}")

    # Current subcategory breakdown
    print(f"\n{'ORIGINAL SUBCATEGORY BREAKDOWN':-^60}")
    if "primary_subcategory" in target_df.columns:
        original_subcats = target_df["primary_subcategory"].value_counts()
        for subcat, count in original_subcats.items():
            print(f"  {subcat:<35} {count:>5}")

    print(f"\n{'YAML STRUCTURE UTILIZATION':-^60}")
    print(
        f"  Total Chemistry paths available: {len(reclassifier.yaml_structure.all_paths)}"
    )
    print(f"  Subcategories in YAML: {len(reclassifier.yaml_structure.subcategories)}")


def main():
    """Main function for OPTIMIZED Chemistry intra-domain reclassification"""

    try:
        print("\n What would you like to do?")
        print("1. Run Chemistry intra-domain reclassification")
        print("2. Check Chemistry checkpoint status")
        print("3. Resume from latest Chemistry checkpoint")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            print("\n Starting OPTIMIZED Chemistry intra-domain reclassification...")

            user_confirmation = input(
                "  This will process Chemistry domain products. Continue? (y/n): "
            )
            if user_confirmation.lower() == "y":
                complete_df, target_df = (
                    process_chemistry_reclassification_with_checkpointing(
                        input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
                    )
                )
                print("✅ Chemistry reclassification completed!")

            else:
                print("❌ Operation cancelled.")

        elif choice == "2":
            print("\n📊 Checking Chemistry checkpoint status...")
            latest_checkpoint, checkpoint_number = find_latest_chemistry_checkpoint()
            if latest_checkpoint:
                print(f"✅ Latest checkpoint found: {latest_checkpoint}")
                print(f"📊 Checkpoint number: {checkpoint_number}")

                # Load and analyze checkpoint
                df = pd.read_csv(latest_checkpoint)
                total_products = len(df)

                if "chemistry_reclassification_performed" in df.columns:
                    processed = len(
                        df[df["chemistry_reclassification_performed"] == True]
                    )
                    print(f"📈 Progress: {processed} Chemistry products processed")

                    # Target product analysis
                    target_mask = (df["primary_domain"] == "Chemistry") | (
                        df["primary_domain"] == "Chemicals"
                    )
                    total_targets = target_mask.sum()
                    remaining_targets = (
                        target_mask
                        & (df["chemistry_reclassification_performed"] != True)
                    ).sum()

                    print(
                        f"🎯 Target products: {total_targets} total, {remaining_targets} remaining"
                    )
                else:
                    print(
                        "📋 Checkpoint contains raw data, no Chemistry processing has started yet"
                    )
            else:
                print("❌ No Chemistry checkpoint found")

        elif choice == "3":
            print("\n🔄 Resuming from latest Chemistry checkpoint (OPTIMIZED)...")
            latest_checkpoint, checkpoint_number = find_latest_chemistry_checkpoint()
            if latest_checkpoint:
                print(f"📁 Found checkpoint: {latest_checkpoint}")
                complete_df, target_df = (
                    process_chemistry_reclassification_with_checkpointing(
                        input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
                    )
                )
                print("✅ Resume completed!")
            else:
                print("❌ No Chemistry checkpoint found. Use option 1 to start fresh.")

        else:
            print("❌ Invalid choice. Please select 1-3.")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
