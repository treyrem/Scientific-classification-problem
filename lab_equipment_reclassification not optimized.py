# LAB EQUIPMENT INTRA-DOMAIN RECLASSIFICATION SYSTEM
# Targets products stuck in generic "Analytical Instrumentation" and "Laboratory Supplies & Consumables"
# Reclassifies them to specific subcategories within Lab Equipment domain using web search + YAML navigation

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

# Configuration
INPUT_CSV = "products_comprehensive_web_search_reclassified_FIXED.csv"
OUTPUT_CSV = "products_lab_equipment_intra_domain_reclassified.csv"

# Lab Equipment YAML file
LAB_EQUIPMENT_YAML = "category_structure_lab_equipment.yaml"

# Checkpointing Configuration
LAB_EQUIPMENT_CHECKPOINT_DIR = "lab_equipment_reclassification_checkpoints"
LAB_EQUIPMENT_CHECKPOINT_FREQ = 25  # Save every N items
LAB_EQUIPMENT_CHECKPOINT_PREFIX = "lab_equipment_reclassification_checkpoint"

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


def find_latest_lab_equipment_checkpoint():
    """Find the most recent lab equipment reclassification checkpoint file"""
    if not os.path.exists(LAB_EQUIPMENT_CHECKPOINT_DIR):
        return None, 0

    checkpoint_files = glob.glob(
        os.path.join(
            LAB_EQUIPMENT_CHECKPOINT_DIR, f"{LAB_EQUIPMENT_CHECKPOINT_PREFIX}_*.csv"
        )
    )
    if not checkpoint_files:
        return None, 0

    # Extract checkpoint numbers and find the latest
    checkpoint_numbers = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        try:
            # Extract number from filename like "lab_equipment_reclassification_checkpoint_150.csv"
            number = int(filename.split("_")[-1].split(".")[0])
            checkpoint_numbers.append((number, filepath))
        except (ValueError, IndexError):
            continue

    if checkpoint_numbers:
        # Return the filepath and number of the highest numbered checkpoint
        latest = max(checkpoint_numbers, key=lambda x: x[0])
        return latest[1], latest[0]

    return None, 0


class LabEquipmentYAMLStructure:
    """Load and navigate the Lab Equipment YAML structure"""

    def __init__(self, yaml_file: str = LAB_EQUIPMENT_YAML):
        self.yaml_file = yaml_file
        self.structure = {}
        self.subcategories = {}
        self.all_paths = []
        self.load_structure()

    def load_structure(self):
        """Load the Lab Equipment YAML structure"""
        try:
            with open(self.yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Extract the Lab Equipment structure
            self.structure = data.get("categories", {}).get("Lab Equipment", {})
            self.subcategories = self.structure.get("subcategories", {})

            # Extract all possible classification paths
            self.all_paths = self._extract_all_paths()

            logger.info(
                f"✅ Loaded Lab Equipment YAML with {len(self.subcategories)} subcategories"
            )
            logger.info(f"✅ Found {len(self.all_paths)} possible classification paths")

        except Exception as e:
            logger.error(f"❌ Failed to load Lab Equipment YAML: {e}")
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
        """Generate a comprehensive prompt showing the Lab Equipment structure"""
        lines = [
            "LAB EQUIPMENT DOMAIN STRUCTURE:",
            "",
            "AVAILABLE SUBCATEGORIES AND CLASSIFICATION PATHS:",
        ]

        # Group paths by depth for clarity
        depth_2_paths = [p for p in self.all_paths if len(p.split(" -> ")) == 2]
        depth_3_paths = [p for p in self.all_paths if len(p.split(" -> ")) == 3]
        depth_4_plus_paths = [p for p in self.all_paths if len(p.split(" -> ")) >= 4]

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
                "✅ Spectrophotometer → Spectroscopy -> UV-Vis Spectrophotometers",
                "✅ Flow cytometer → Cell Analysis -> Flow Cytometers",
                "✅ Centrifuge → General Laboratory Equipment -> Centrifuges",
                "✅ Whatman filter → Laboratory Supplies & Consumables -> Filters",
                "✅ PCR plate → Laboratory Supplies & Consumables -> PCR Consumables -> 96-Well PCR Plates",
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
        if path_components[0] in self.subcategories:
            return True, path_components[0]

        return False, f"Invalid path: {test_path}"


class LabEquipmentIntraDomainReclassifier:
    """Reclassify Lab Equipment products from generic to specific subcategories"""

    def __init__(self):
        self.yaml_structure = LabEquipmentYAMLStructure()
        self.search_cache = {}
        self.total_searches = 0
        self.successful_searches = 0
        self.reclassification_stats = defaultdict(int)

    def identify_target_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify Lab Equipment products stuck in generic subcategories"""

        logger.info(
            "🎯 Identifying Lab Equipment products for intra-domain reclassification..."
        )

        # Target conditions
        lab_equipment_mask = (df["primary_domain"] == "Lab_Equipment") | (
            df["primary_domain"] == "Lab Equipment"
        )

        generic_subcategory_mask = (
            df["primary_subcategory"] == "Analytical Instrumentation"
        ) | (df["primary_subcategory"] == "Laboratory Supplies & Consumables")

        # Combine conditions
        target_mask = lab_equipment_mask & generic_subcategory_mask

        target_df = df[target_mask].copy()

        logger.info(
            f"📊 Found {len(target_df)} Lab Equipment products in generic subcategories:"
        )

        if len(target_df) > 0:
            subcategory_counts = target_df["primary_subcategory"].value_counts()
            for subcat, count in subcategory_counts.items():
                logger.info(f"  - {subcat}: {count} products")

        return target_df

    def reclassify_lab_equipment_product(
        self,
        product_name: str,
        manufacturer: str = "",
        current_subcategory: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """Reclassify a single Lab Equipment product to specific subcategory"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 LAB EQUIPMENT RECLASSIFICATION: '{product_name}'")
        logger.info(f"Current: Lab Equipment → {current_subcategory}")
        logger.info(f"{'='*60}")

        total_tokens = 0

        # Step 1: Enhanced web search for product details
        logger.info("🔍 STEP 1: Enhanced Product Research")
        search_result = self.enhanced_product_search(product_name, manufacturer)
        total_tokens += search_result.get("token_usage", 0)

        if not search_result.get("search_successful", False):
            logger.warning(f"⚠️ Web search failed for '{product_name}'")
            return self._create_failed_search_result(
                product_name, current_subcategory, total_tokens
            )

        # Step 2: Lab Equipment specific classification
        logger.info("🗂️ STEP 2: Lab Equipment Specific Classification")
        enhanced_description = search_result.get("enhanced_description", description)

        classification_result = self.classify_within_lab_equipment(
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
                "Lab Equipment classification failed",
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
        cache_key = f"lab_equipment_{search_query.lower()}"
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached results for: {search_query}")
            return self.search_cache[cache_key]

        logger.info(f"🔍 Enhanced product search for: {search_query}")
        self.total_searches += 1

        prompt = f"""You are a laboratory equipment specialist. Research this product and provide comprehensive technical details for accurate classification.

PRODUCT TO RESEARCH: "{product_name}" by {manufacturer}

CRITICAL RESEARCH FOCUS - Provide detailed information for EACH category:

1. INSTRUMENT TYPE IDENTIFICATION:
   - Is this a spectrophotometer, spectrometer, or spectroscopy instrument?
   - Is this a flow cytometer, cell analyzer, or cell sorting instrument?  
   - Is this a centrifuge (benchtop, microcentrifuge, ultracentrifuge)?
   - Is this a particle size analyzer, particle counter?
   - Is this an HPLC, GC, or chromatography system?
   - Is this a PCR machine, thermocycler, qPCR system?
   - Is this a microscope, imaging system?
   - Is this a mass spectrometer?

2. CONSUMABLE/SUPPLY TYPE IDENTIFICATION:
   - Is this a filter (membrane filter, syringe filter, Whatman filter)?
   - Is this a membrane (PVDF, nitrocellulose, nylon)?
   - Is this a plate (PCR plate, microplate, cell culture plate)?
   - Is this a tube (PCR tube, microcentrifuge tube, cryogenic tube)?
   - Is this a pipette tip, syringe, or liquid handling consumable?

3. TECHNICAL SPECIFICATIONS:
   - Wavelength range (for spectroscopy equipment)
   - Detection method (UV-Vis, fluorescence, chemiluminescence)
   - Capacity and speed (for centrifuges)
   - Flow rate and pressure (for chromatography)
   - Laser specifications (for flow cytometry)
   - Pore size and material (for filters)
   - Volume and format (for consumables)

4. PRIMARY APPLICATION AREAS:
   - Cell analysis, cell sorting, flow cytometry
   - Spectroscopy, spectrophotometry, absorbance measurement
   - Chromatography, separation, purification
   - PCR, amplification, thermal cycling
   - Microscopy, imaging, visualization
   - Sample preparation, liquid handling

5. EXACT PRODUCT CATEGORY (be very specific):
   - For instruments: What exact type of analytical instrument?
   - For consumables: What exact type of lab supply/consumable?

RESPONSE FORMAT (JSON only):
{{
    "instrument_type": "exact_instrument_type_if_applicable",
    "consumable_type": "exact_consumable_type_if_applicable", 
    "primary_function": "detailed_primary_function_and_use",
    "technical_specifications": "key_technical_details_and_specs",
    "detection_method": "detection_or_measurement_method",
    "application_areas": ["specific_lab_applications"],
    "product_category": "most_specific_equipment_category",
    "enhanced_description": "comprehensive_technical_description_for_classification",
    "search_successful": true
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,  # Lower temperature for more factual responses
                max_tokens=1200,  # More tokens for comprehensive info
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

            logger.info(f"✅ Enhanced product research completed")
            logger.info(f"   Instrument Type: {result.get('instrument_type', 'N/A')}")
            logger.info(f"   Consumable Type: {result.get('consumable_type', 'N/A')}")
            logger.info(f"   Category: {result.get('product_category', 'Unknown')}")

            time.sleep(0.2)  # Rate limiting
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return {"search_successful": False, "token_usage": 0}
        except Exception as e:
            logger.error(f"❌ Enhanced product search failed: {e}")
            return {"search_successful": False, "token_usage": 0}

    def classify_within_lab_equipment(
        self, product_name: str, description: str, current_subcategory: str
    ) -> Optional[Dict]:
        """Classify product within Lab Equipment domain using STRICT YAML structure - NO HALLUCINATIONS"""

        logger.info(f"🗂️ STRICT YAML-based classification within Lab Equipment domain")

        # Get the complete Lab Equipment structure
        structure_prompt = self.yaml_structure.get_structure_prompt()

        prompt = f"""You are a laboratory equipment classification expert. Classify this Lab Equipment product using ONLY the EXACT paths from the YAML structure below.

{structure_prompt}

PRODUCT TO CLASSIFY:
Name: "{product_name}"
Enhanced Description: "{description}"
Current Classification: Lab Equipment → {current_subcategory}

STRICT CLASSIFICATION RULES - NO HALLUCINATIONS:

1. **ONLY USE EXACT NAMES** from the classification paths shown above
2. **DO NOT CREATE** new subcategory names or modify existing ones
3. **GO AS DEEP AS POSSIBLE** - aim for 3+ levels when the YAML structure supports it
4. **MATCH KEYWORDS** to find the most specific classification:

KEYWORD MATCHING GUIDE:
- "spectrophotometer", "spectrometer", "UV-Vis" → Spectroscopy → UV-Vis Spectrophotometers
- "flow cytometer", "cytometer", "FACS" → Cell Analysis → Flow Cytometers  
- "centrifuge", "ultracentrifuge" → General Laboratory Equipment → Centrifuges
- "particle size", "particle analyzer" → Analytical Instrumentation → Particle Size Analyzer
- "filter", "membrane filter", "whatman" → Laboratory Supplies & Consumables → Filters
- "PCR plate", "96-well PCR" → Laboratory Supplies & Consumables → PCR Consumables → 96-Well PCR Plates
- "microplate", "96-well", "384-well" → Laboratory Supplies & Consumables → 96-Well Microplates
- "HPLC", "liquid chromatography" → Chromatography Equipment → High Performance Liquid Chromatography (HPLC)
- "mass spectrometer", "LC-MS" → Mass Spectrometry → Mass Spectrometers
- "qPCR", "real-time PCR", "thermocycler" → PCR Equipment → PCR Thermocyclers → Real-Time PCR Systems
- "microplate reader", "plate reader" → Spectroscopy → Microplate Readers → Multi-Mode Readers

5. **DEPTH REQUIREMENTS** (GO DEEPER WHEN POSSIBLE):
   - MINIMUM: subcategory (level 1)
   - STRONGLY PREFERRED: subsubcategory (level 2) 
   - OPTIMAL: subsubsubcategory (level 3) when YAML structure supports it
   - MAXIMUM: Use deepest level available in YAML

6. **VALIDATION**: Your response will be validated against the YAML structure - invalid paths will be rejected

EXAMPLES OF GOING DEEPER:
❌ SHALLOW: "Spectroscopy" (only 1 level)
✅ BETTER: "Spectroscopy → UV-Vis Spectrophotometers" (2 levels)
✅ BEST: "Spectroscopy → Microplate Readers → Multi-Mode Readers" (3 levels)

Respond with JSON only:
{{
    "subcategory": "EXACT_subcategory_name_from_YAML_paths",
    "subsubcategory": "EXACT_subsubcategory_name_from_YAML_or_null",
    "subsubsubcategory": "EXACT_subsubsubcategory_name_from_YAML_or_null",
    "confidence": "High/Medium/Low"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,  # Very low temperature to reduce hallucinations
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
                    "validated_path": f"Lab Equipment -> {validated_path}",
                    "depth_achieved": len(path_components),
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            logger.info(f"✅ STRICT YAML classification successful: {validated_path}")
            logger.info(f"   Depth achieved: {len(path_components)} levels")

            # Log if we could have gone deeper
            max_possible_depth = max(
                [
                    len(p.split(" -> "))
                    for p in self.yaml_structure.all_paths
                    if p.startswith(path_components[0])
                ]
                + [0]
            )
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
            "lab_equipment_reclassification_performed": True,
            "lab_equipment_reclassification_successful": True,
            "old_subcategory": old_subcategory,
            # New classification data to update primary_* columns
            "new_subcategory": classification.get("subcategory", ""),
            "new_subsubcategory": classification.get("subsubcategory", ""),
            "new_subsubsubcategory": classification.get("subsubsubcategory", ""),
            "new_confidence": classification.get("confidence", ""),
            "new_validated_path": classification.get("validated_path", ""),
            "depth_achieved": classification.get("depth_achieved", 0),
            "total_tokens": tokens,
            "classification_method": "lab_equipment_intra_domain_reclassification",
        }

    def _create_failed_search_result(
        self, product_name: str, current_subcategory: str, tokens: int
    ) -> Dict[str, Any]:
        """Create result for failed web search"""
        return {
            "lab_equipment_reclassification_performed": True,
            "lab_equipment_reclassification_successful": False,
            "old_subcategory": current_subcategory,
            "total_tokens": tokens,
            "classification_method": "failed_lab_equipment_search",
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
            "lab_equipment_reclassification_performed": True,
            "lab_equipment_reclassification_successful": False,
            "old_subcategory": current_subcategory,
            "total_tokens": tokens,
            "classification_method": "failed_lab_equipment_classification",
            "failure_reason": reason,
        }

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def get_lab_equipment_reclassification_columns():
    """Get minimal tracking columns for Lab Equipment reclassification"""
    return [
        # Minimal tracking only - update existing primary_* columns directly
        "lab_equipment_reclassification_performed",
        "lab_equipment_reclassification_successful",
        "lab_equipment_original_subcategory",  # Track what it was before
        "lab_equipment_processed_at",
    ]


def process_lab_equipment_reclassification_with_checkpointing(
    input_csv: str, output_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process Lab Equipment intra-domain reclassification with checkpointing"""

    logger.info(
        "🔧 Starting Lab Equipment Intra-Domain Reclassification with Checkpointing..."
    )

    # Setup checkpoint directory
    if os.path.exists(LAB_EQUIPMENT_CHECKPOINT_DIR) and not os.path.isdir(
        LAB_EQUIPMENT_CHECKPOINT_DIR
    ):
        logger.error(
            f"'{LAB_EQUIPMENT_CHECKPOINT_DIR}' exists but is not a directory. Please remove or rename this file."
        )
        exit(1)
    os.makedirs(LAB_EQUIPMENT_CHECKPOINT_DIR, exist_ok=True)

    # Initialize reclassifier
    reclassifier = LabEquipmentIntraDomainReclassifier()

    try:
        # Check for existing checkpoint
        latest_checkpoint, last_checkpoint_number = (
            find_latest_lab_equipment_checkpoint()
        )

        if latest_checkpoint:
            logger.info(f"📁 Found Lab Equipment checkpoint: {latest_checkpoint}")
            logger.info(f"📊 Last checkpoint number: {last_checkpoint_number}")
            complete_df = pd.read_csv(latest_checkpoint, low_memory=False)

            # Find unprocessed products
            already_processed_count = 0
            if "lab_equipment_reclassification_performed" in complete_df.columns:
                already_processed_count = (
                    complete_df["lab_equipment_reclassification_performed"] == True
                ).sum()
                logger.info(
                    f"📊 Found {already_processed_count} products already processed"
                )

            # Find target products that haven't been processed
            target_mask = (
                (
                    (complete_df["primary_domain"] == "Lab_Equipment")
                    | (complete_df["primary_domain"] == "Lab Equipment")
                )
                & (
                    (complete_df["primary_subcategory"] == "Analytical Instrumentation")
                    | (
                        complete_df["primary_subcategory"]
                        == "Laboratory Supplies & Consumables"
                    )
                )
                & (complete_df["lab_equipment_reclassification_performed"] != True)
            )

            target_df = complete_df[target_mask].copy()
            logger.info(f"📊 Unprocessed target products found: {len(target_df)}")

        else:
            # Start fresh - load complete dataset
            logger.info(f"🆕 No checkpoint found, starting fresh from {input_csv}")
            complete_df = pd.read_csv(input_csv)
            logger.info(f"✅ Loaded {len(complete_df)} total products")

            # Initialize minimal Lab Equipment reclassification columns
            lab_equipment_columns = get_lab_equipment_reclassification_columns()
            for col in lab_equipment_columns:
                if col.endswith("_performed") or col.endswith("_successful"):
                    complete_df[col] = False
                else:
                    complete_df[col] = ""

            # Identify target products
            target_df = reclassifier.identify_target_products(complete_df)
            already_processed_count = 0

        if len(target_df) == 0:
            logger.info(
                "✅ No target Lab Equipment products found for reclassification!"
            )
            return complete_df, pd.DataFrame()

        total_target_products = len(target_df)
        logger.info(f"📊 Processing {total_target_products} Lab Equipment products...")
        logger.info(f"📈 Total progress so far: {already_processed_count} processed")

        # Process each target product
        successful_reclassifications = 0
        failed_reclassifications = 0

        progress_desc = (
            f"🔧 Lab Equipment Reclassification ({already_processed_count} done)"
        )

        for i, (idx, row) in enumerate(
            tqdm(
                target_df.iterrows(),
                desc=progress_desc,
                total=len(target_df),
                initial=0,
            )
        ):
            name = row["Name"]
            manufacturer = row.get("Manufacturer", "") if "Manufacturer" in row else ""
            current_subcategory = row.get("primary_subcategory", "")
            description = row.get("Description", "") if "Description" in row else ""

            try:
                # Perform Lab Equipment reclassification
                result = reclassifier.reclassify_lab_equipment_product(
                    name,
                    manufacturer,
                    current_subcategory,
                    description,
                )

                # Store results in the complete dataframe
                complete_df.at[idx, "lab_equipment_original_subcategory"] = (
                    current_subcategory
                )
                complete_df.at[idx, "lab_equipment_reclassification_performed"] = (
                    result.get("lab_equipment_reclassification_performed", False)
                )
                complete_df.at[idx, "lab_equipment_reclassification_successful"] = (
                    result.get("lab_equipment_reclassification_successful", False)
                )

                # Update existing primary_* columns directly if successful
                if result.get("lab_equipment_reclassification_successful", False):
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
                    logger.info(
                        f"✅ Successfully reclassified: {name[:30]} → {new_subcategory}"
                    )
                    if new_subsubcategory:
                        logger.info(
                            f"   Deep classification: {new_subcategory} → {new_subsubcategory}"
                        )
                else:
                    failed_reclassifications += 1

                # Store processing metadata
                complete_df.at[idx, "lab_equipment_processed_at"] = (
                    datetime.now().isoformat()
                )

            except Exception as e:
                logger.error(f"❌ Error processing '{name}': {e}")
                complete_df.at[idx, "lab_equipment_reclassification_performed"] = True
                complete_df.at[idx, "lab_equipment_reclassification_successful"] = False
                complete_df.at[idx, "lab_equipment_processed_at"] = (
                    datetime.now().isoformat()
                )
                failed_reclassifications += 1

            # Save checkpoint
            if (i + 1) % LAB_EQUIPMENT_CHECKPOINT_FREQ == 0 or i == len(target_df) - 1:
                actual_progress = already_processed_count + i + 1
                checkpoint_file = os.path.join(
                    LAB_EQUIPMENT_CHECKPOINT_DIR,
                    f"{LAB_EQUIPMENT_CHECKPOINT_PREFIX}_{actual_progress}.csv",
                )
                complete_df.to_csv(checkpoint_file, index=False)
                logger.info(f"💾 Saved Lab Equipment checkpoint: {checkpoint_file}")
                logger.info(f"📈 Total progress: {actual_progress} products processed")

        # Save final result
        complete_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Complete Lab Equipment reclassification saved to {output_csv}")

        # Generate report
        generate_lab_equipment_reclassification_report(
            target_df,
            reclassifier,
            successful_reclassifications,
            failed_reclassifications,
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"🔧 LAB EQUIPMENT RECLASSIFICATION SUMMARY")
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
        logger.error(f"❌ Error in Lab Equipment reclassification: {e}")
        raise


def generate_lab_equipment_reclassification_report(
    target_df: pd.DataFrame,
    reclassifier: LabEquipmentIntraDomainReclassifier,
    successful: int,
    failed: int,
):
    """Generate report for Lab Equipment reclassification results"""

    print(f"\n{'='*80}")
    print("🔧 LAB EQUIPMENT INTRA-DOMAIN RECLASSIFICATION REPORT")
    print(f"{'='*80}")

    total = len(target_df)

    print(f"📊 OVERVIEW")
    print(f"  Target Lab Equipment products processed: {total}")
    print(f"  Successful reclassifications: {successful} ({successful/total*100:.1f}%)")
    print(f"  Failed reclassifications: {failed} ({failed/total*100:.1f}%)")

    # Search statistics
    print(f"\n{'SEARCH STATISTICS':-^60}")
    print(f"  Total searches performed: {reclassifier.total_searches}")
    print(f"  Successful searches: {reclassifier.successful_searches}")
    if reclassifier.total_searches > 0:
        print(
            f"  Search success rate: {reclassifier.successful_searches/reclassifier.total_searches*100:.1f}%"
        )

    # Current subcategory breakdown
    print(f"\n{'ORIGINAL SUBCATEGORY BREAKDOWN':-^60}")
    if "primary_subcategory" in target_df.columns:
        original_subcats = target_df["primary_subcategory"].value_counts()
        for subcat, count in original_subcats.items():
            print(f"  {subcat:<35} {count:>5}")

    print(f"\n{'YAML STRUCTURE UTILIZATION':-^60}")
    print(
        f"  Total Lab Equipment paths available: {len(reclassifier.yaml_structure.all_paths)}"
    )
    print(f"  Subcategories in YAML: {len(reclassifier.yaml_structure.subcategories)}")

    print(f"\n{'EXPECTED IMPROVEMENTS':-^60}")
    print(f"  Products should move from generic subcategories to:")
    print(f"  - Spectroscopy → UV-Vis Spectrophotometers (for spectrophotometers)")
    print(f"  - Cell Analysis → Flow Cytometers (for flow cytometers)")
    print(f"  - General Laboratory Equipment → Centrifuges (for centrifuges)")
    print(f"  - Laboratory Supplies & Consumables → Filters (for filters)")
    print(f"  - PCR Equipment → PCR Thermocyclers → Real-Time PCR Systems")
    print(f"  - Chromatography Equipment → HPLC → specific HPLC types")

    print(f"\n{'DEPTH ANALYSIS':-^60}")
    print(f"  Target: Achieve 2-3 levels of classification depth")
    print(f"  Goal: Move from generic 1-level to specific 2-3 level classifications")


def main():
    """Main function for Lab Equipment intra-domain reclassification"""
    print("=" * 80)
    print("🔧 LAB EQUIPMENT INTRA-DOMAIN RECLASSIFICATION")
    print("Reclassifies generic Lab Equipment subcategories to specific ones")
    print("=" * 80)

    try:
        print("\n🎯 What would you like to do?")
        print("1. Run Lab Equipment intra-domain reclassification")
        print("2. Check Lab Equipment checkpoint status")
        print("3. Resume from latest Lab Equipment checkpoint")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            print("\n🔧 Starting Lab Equipment intra-domain reclassification...")
            user_confirmation = input(
                "⚠️  This will process Lab Equipment products. Continue? (y/n): "
            )
            if user_confirmation.lower() == "y":
                complete_df, target_df = (
                    process_lab_equipment_reclassification_with_checkpointing(
                        input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
                    )
                )
                print("✅ Lab Equipment reclassification completed!")
            else:
                print("❌ Operation cancelled.")

        elif choice == "2":
            print("\n📊 Checking Lab Equipment checkpoint status...")
            latest_checkpoint, checkpoint_number = (
                find_latest_lab_equipment_checkpoint()
            )
            if latest_checkpoint:
                print(f"✅ Latest checkpoint found: {latest_checkpoint}")
                print(f"📊 Checkpoint number: {checkpoint_number}")

                # Load and analyze checkpoint
                df = pd.read_csv(latest_checkpoint)
                total_products = len(df)

                if "lab_equipment_reclassification_performed" in df.columns:
                    processed = len(
                        df[df["lab_equipment_reclassification_performed"] == True]
                    )
                    print(f"📈 Progress: {processed} Lab Equipment products processed")

                    # Target product analysis
                    target_mask = (
                        (df["primary_domain"] == "Lab_Equipment")
                        | (df["primary_domain"] == "Lab Equipment")
                    ) & (
                        (df["primary_subcategory"] == "Analytical Instrumentation")
                        | (
                            df["primary_subcategory"]
                            == "Laboratory Supplies & Consumables"
                        )
                    )
                    total_targets = target_mask.sum()
                    remaining_targets = (
                        target_mask
                        & (df["lab_equipment_reclassification_performed"] != True)
                    ).sum()

                    print(
                        f"🎯 Target products: {total_targets} total, {remaining_targets} remaining"
                    )
                else:
                    print(
                        "📋 Checkpoint contains raw data, no Lab Equipment processing has started yet"
                    )
            else:
                print("❌ No Lab Equipment checkpoint found")

        elif choice == "3":
            print("\n🔄 Resuming from latest Lab Equipment checkpoint...")
            latest_checkpoint, checkpoint_number = (
                find_latest_lab_equipment_checkpoint()
            )
            if latest_checkpoint:
                print(f"📁 Found checkpoint: {latest_checkpoint}")
                complete_df, target_df = (
                    process_lab_equipment_reclassification_with_checkpointing(
                        input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
                    )
                )
                print("✅ Resume completed!")
            else:
                print(
                    "❌ No Lab Equipment checkpoint found. Use option 1 to start fresh."
                )

        else:
            print("❌ Invalid choice. Please select 1-3.")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
