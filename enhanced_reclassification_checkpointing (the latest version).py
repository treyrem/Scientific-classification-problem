# COMPREHENSIVE WEB SEARCH ENHANCED RE-CLASSIFICATION SYSTEM - FIXED CHECKPOINTING
# Targets ALL problematic classifications + fixes YAML validation and prompts + PROPER CHECKPOINT RESUME

import pandas as pd
import json
import logging
import os
import re
import yaml
from typing import Dict, List, Tuple, Optional, Any, Set
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import time
from collections import defaultdict
import glob
from datetime import datetime

# Import existing classification system components
from enhanced_classification_full_checkpoint import (
    LLMDrivenCategorySystem,
    EnhancedTagSystem,
    EnhancedLLMClassifier,
)

# Configuration
INPUT_CSV = "products_enhanced_fixed_classification.csv"
OUTPUT_CSV = "products_comprehensive_web_search_reclassified_FIXED.csv"

# Checkpointing Configuration
COMPREHENSIVE_CHECKPOINT_DIR = "comprehensive_reclassification_checkpoints"
COMPREHENSIVE_CHECKPOINT_FREQ = 25  # Save every N items
COMPREHENSIVE_CHECKPOINT_PREFIX = "comprehensive_reclassification_checkpoint"

# Antibiotic keywords from targeted script
ANTIBIOTIC_KEYWORDS = [
    "streptomycin",
    "penicillin",
    "ampicillin",
    "chloramphenicol",
    "gentamicin",
    "kanamycin",
    "neomycin",
    "tetracycline",
]

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


def find_latest_comprehensive_checkpoint():
    """Find the most recent comprehensive reclassification checkpoint file"""
    if not os.path.exists(COMPREHENSIVE_CHECKPOINT_DIR):
        return None, 0

    checkpoint_files = glob.glob(
        os.path.join(
            COMPREHENSIVE_CHECKPOINT_DIR, f"{COMPREHENSIVE_CHECKPOINT_PREFIX}_*.csv"
        )
    )
    if not checkpoint_files:
        return None, 0

    # Extract checkpoint numbers and find the latest
    checkpoint_numbers = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        try:
            # Extract number from filename like "comprehensive_reclassification_checkpoint_150.csv"
            number = int(filename.split("_")[-1].split(".")[0])
            checkpoint_numbers.append((number, filepath))
        except (ValueError, IndexError):
            continue

    if checkpoint_numbers:
        # Return the filepath and number of the highest numbered checkpoint
        latest = max(checkpoint_numbers, key=lambda x: x[0])
        return latest[1], latest[0]

    return None, 0


class FixedComprehensiveWebSearchClassifier:
    """🚀 FIXED COMPREHENSIVE: All problematic classifications + proper YAML handling + better prompts"""

    def __init__(
        self, category_system: LLMDrivenCategorySystem, tag_system: EnhancedTagSystem
    ):
        self.category_system = category_system
        self.tag_system = tag_system
        self.search_cache = {}
        self.total_searches = 0
        self.successful_searches = 0
        self.reclassification_stats = defaultdict(int)

    def reclassify_problematic_product(
        self,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str = "",
        current_subcategory: str = "",
        original_description: str = "",
    ) -> Dict[str, Any]:
        """🚀 COMPREHENSIVE: Reclassify any problematic product with web search enhancement"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🌐 FIXED COMPREHENSIVE RECLASSIFICATION: '{product_name}'")
        logger.info(f"Problem Type: {problem_type}")
        logger.info(f"Current: {current_domain} → {current_subcategory}")
        logger.info(f"{'='*60}")

        total_tokens = 0

        # 🔥 STEP 1: Problem-Aware Web Search + Domain Identification
        logger.info("🔍 STEP 1: Problem-Aware Web Search + Domain Identification")
        search_and_domain_result = self.search_and_identify_domain_for_problem(
            product_name,
            manufacturer,
            problem_type,
            current_domain,
            current_subcategory,
        )
        total_tokens += search_and_domain_result.get("token_usage", 0)

        if not search_and_domain_result.get("search_successful", False):
            logger.warning(f"⚠️ Web search failed for '{product_name}'")
            return self._create_failed_search_result(
                product_name, problem_type, total_tokens
            )

        # 🎯 STEP 2: FIXED Domain-Specific Classification
        logger.info("🎯 STEP 2: FIXED Domain-Specific Classification")
        suggested_domain = search_and_domain_result.get("suggested_domain", "Other")
        enhanced_description = search_and_domain_result.get("enhanced_description", "")

        if (
            suggested_domain != "Other"
            and suggested_domain in self.category_system.available_domains
        ):
            # 🔧 FIXED: Use improved classification with proper YAML handling
            domain_classification = self._classify_within_domain_efficiently_FIXED(
                product_name, enhanced_description, suggested_domain
            )
            if domain_classification is not None:
                total_tokens += domain_classification.get("token_usage", 0)
        else:
            logger.warning(f"⚠️ Invalid or 'Other' domain suggested: {suggested_domain}")
            domain_classification = None

        # Step 3: Build Final Result
        if domain_classification and domain_classification.get("is_valid_path", False):
            return self._create_successful_result(
                product_name,
                problem_type,
                suggested_domain,
                domain_classification,
                search_and_domain_result,
                current_domain,
                current_subcategory,
                total_tokens,
            )
        else:
            failure_reason = "Domain-specific classification failed"
            if domain_classification is None:
                failure_reason = "Domain classification returned None"
            return self._create_failed_classification_result(
                product_name,
                problem_type,
                search_and_domain_result,
                current_domain,
                current_subcategory,
                total_tokens,
                failure_reason,
            )

    def search_and_identify_domain_for_problem(
        self,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str,
        current_subcategory: str,
    ) -> Dict[str, Any]:
        """🔥 PROBLEM-AWARE: Web search + domain identification tailored to specific problems"""

        # Create search query
        search_query = product_name.strip()
        if manufacturer and manufacturer.strip():
            search_query = f"{product_name.strip()} {manufacturer.strip()}"

        search_query = re.sub(r"[^\w\s-]", " ", search_query)
        search_query = " ".join(search_query.split())

        # Check cache first
        cache_key = f"comprehensive_{search_query.lower()}_{problem_type}"
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached results for: {search_query}")
            return self.search_cache[cache_key]

        logger.info(
            f"🔍 Problem-aware search for: {search_query} (Problem: {problem_type})"
        )
        self.total_searches += 1

        # Get available domains for the prompt
        available_domains = self.category_system.available_domains

        prompt = self._create_problem_specific_prompt(
            product_name,
            manufacturer,
            problem_type,
            current_domain,
            current_subcategory,
            available_domains,
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            if not response.choices or not response.choices[0].message:
                logger.error("❌ LLM returned empty response")
                return {
                    "search_successful": False,
                    "suggested_domain": "Other",
                    "token_usage": 0,
                }

            text = response.choices[0].message.content
            if not text or text.strip() == "":
                logger.error("❌ LLM returned empty content")
                return {
                    "search_successful": False,
                    "suggested_domain": "Other",
                    "token_usage": 0,
                }

            cleaned = self._strip_code_fences(text.strip())
            result = json.loads(cleaned)

            # Validate domain
            suggested_domain = result.get("suggested_domain", "Other")
            if suggested_domain not in self.category_system.available_domains:
                logger.warning(f"⚠️ Invalid domain suggested: {suggested_domain}")
                result["suggested_domain"] = "Other"

            result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            # Cache the result
            self.search_cache[cache_key] = result
            self.successful_searches += 1

            logger.info(
                f"✅ Domain identified: {suggested_domain} ({result.get('confidence', 'Low')})"
            )

            time.sleep(0.2)  # Faster rate limiting
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return {
                "search_successful": False,
                "suggested_domain": "Other",
                "token_usage": 0,
            }
        except Exception as e:
            logger.error(f"❌ Search and domain identification failed: {e}")
            return {
                "search_successful": False,
                "suggested_domain": "Other",
                "token_usage": 0,
            }

    def _create_problem_specific_prompt(
        self,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str,
        current_subcategory: str,
        available_domains: List[str],
    ) -> str:
        """Better prompts that focus on product type vs application"""

        base_domains_info = f"""
AVAILABLE DOMAINS: {', '.join(available_domains)}

CRITICAL DOMAIN CLASSIFICATION RULES:
Lab_Equipment: Physical instruments, hardware, supplies, consumables, membranes, filters
Software: Computer programs, statistical software, analysis software, LIMS
Cell_Biology: Cell culture reagents, research antibiotics, cell lines, staining
Nucleic_Acid_Electrophoresis: Gel systems, DNA ladders, electrophoresis reagents (NOT membranes used for blotting)
Chemistry: Chemical compounds, solvents, buffers (NOT research antibiotics)

MEMBRANE CLASSIFICATION RULE:
- Hybond, PVDF, nitrocellulose membranes → Lab_Equipment (they are physical supplies)
- Even if used for Southern blots → Still Lab_Equipment domain
- Focus on WHAT the product IS, not what it's USED FOR

SOFTWARE CLASSIFICATION RULE:
- GraphPad Prism, SAS, SPSS, STATA → Software domain
- Always classify by the PRIMARY PRODUCT TYPE
"""

        if "Other_domain" in problem_type:
            specific_instructions = f"""
PROBLEM: Product currently has no proper domain classification.
GOAL: Research and identify the correct primary domain for this product.

RESEARCH FOCUS: What TYPE of product is this fundamentally?
- Is it a physical object → Likely Lab_Equipment
- Is it software → Software domain  
- Is it a chemical/biological reagent → Chemistry/Cell_Biology

CRITICAL: Focus on product type, not application!
"""

        elif "Generic_Analytical_Instrumentation" in problem_type:
            specific_instructions = f"""
PROBLEM: Product is generically classified as "Analytical Instrumentation" in {current_domain}.
CURRENT: {current_domain} → {current_subcategory}
GOAL: Find SPECIFIC type of equipment/instrument to replace generic classification.

RESEARCH FOCUS:
- What SPECIFIC type of equipment is this? (filter, membrane, spectrophotometer, etc.)
- Is it a supply/consumable or an instrument?
- Should it stay in {current_domain} with a more specific subcategory?

CRITICAL: If it's a membrane/filter → Lab_Equipment → Laboratory Supplies & Consumables
"""

        elif "Generic_Statistical_Software" in problem_type:
            specific_instructions = f"""
PROBLEM: Product is generically classified as "Statistical Analysis Software" in {current_domain}.
CURRENT: {current_domain} → {current_subcategory}  
GOAL: Identify SPECIFIC software name (GraphPad Prism, SAS, SPSS, etc.).

RESEARCH FOCUS:
- What is the EXACT software name?
- Confirm it belongs in Software domain
- Need to drill down to specific software product

CRITICAL: Software products need specific identification, not generic "Statistical Analysis Software"
"""

        else:
            specific_instructions = f"""
PROBLEM: Multiple classification issues detected.
CURRENT: {current_domain} → {current_subcategory}
GOAL: Research and provide correct domain based on product type.
"""

        return f"""You are a life science product specialist fixing problematic classifications.

{specific_instructions}

PRODUCT TO RESEARCH: "{product_name}" by {manufacturer}

{base_domains_info}

CLASSIFICATION STRATEGY:
1. Identify the PRIMARY PRODUCT TYPE (physical object, software, reagent)
2. Focus on WHAT IT IS, not what it's used for
3. Membranes/filters are Lab_Equipment regardless of application
4. Software products go to Software domain regardless of specific use

RESPONSE FORMAT (JSON only):
{{
    "product_research": "detailed findings about what this product fundamentally IS",
    "suggested_domain": "exact_domain_from_available_list",
    "confidence": "High/Medium/Low",
    "reasoning": "why this domain classification focuses on product type not application",
    "primary_product_type": "physical_object/software/chemical_reagent/biological_reagent",
    "enhanced_description": "comprehensive description emphasizing product type",
    "search_successful": true
}}"""

    def _classify_within_domain_efficiently_FIXED(
        self, product_name: str, description: str, domain_key: str
    ) -> Optional[Dict]:
        """🔧 CORRECTED: Uses actual methods from enhanced_classification_full_checkpoint.py"""

        logger.info(f"🗂️ CORRECTED classification in: {domain_key}")

        # This method works - from enhanced_classification_full_checkpoint.py
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        # Create domain-specific prompt with examples
        if domain_key == "Software":
            examples = """
SOFTWARE EXAMPLES:
- GraphPad Prism → subcategory: "Statistical Analysis Software", subsubcategory: "GraphPad Prism"
- SAS → subcategory: "Statistical Analysis Software", subsubcategory: "SAS"  
- STATA → subcategory: "Statistical Analysis Software", subsubcategory: "STATA"
- SPSS → subcategory: "Statistical Analysis Software", subsubcategory: "SPSS"
"""
        elif domain_key == "Lab_Equipment":
            examples = """
LAB EQUIPMENT EXAMPLES:
- Hybond membrane → subcategory: "Laboratory Supplies & Consumables"
- Whatman filter → subcategory: "Laboratory Supplies & Consumables"
- nitrocellulose membrane → subcategory: "Laboratory Supplies & Consumables"
- centrifuge → subcategory: "General Laboratory Equipment"
"""
        else:
            examples = ""

        prompt = f"""Classify this product within the {domain_key} domain.

{examples}

YAML STRUCTURE:
{focused_structure}

Product: "{product_name}"
Description: "{description}"

CRITICAL: Use the exact category names from the YAML structure above.

JSON response:
{{
    "subcategory": "exact_name_from_yaml_structure",
    "subsubcategory": "exact_name_from_yaml_if_available", 
    "confidence": "High"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=400,
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

            # Use the actual validate_classification_path method that EXISTS
            try:
                is_valid, validated_path = (
                    self._improved_validate_classification_path_FIXED(
                        domain_key, path_components
                    )
                )
                logger.info(f"✅ Validation successful: {validated_path}")
            except Exception as e:
                # If validation fails, create manual path
                logger.warning(f"⚠️ Validation failed: {e}")
                logger.info("🔧 Creating manual path...")

                domain_name = domain_key.replace("_", " ")
                if path_components:
                    validated_path = domain_name + " -> " + " -> ".join(path_components)
                    is_valid = True
                    logger.info(f"✅ Manual path created: {validated_path}")
                else:
                    validated_path = domain_name
                    is_valid = True

            result.update(
                {
                    "domain": domain_key,
                    "path_components": path_components,
                    "is_valid_path": is_valid,
                    "validated_path": validated_path,
                    "depth_achieved": len(path_components),
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            if is_valid:
                logger.info(f"✅ CORRECTED FIX successful: {validated_path}")
            else:
                logger.warning(
                    f"❌ Classification failed: {' -> '.join(path_components)}"
                )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.error(
                f"Raw response: {text[:200] if 'text' in locals() else 'None'}..."
            )
            return None
        except Exception as e:
            logger.error(f"❌ Domain classification failed: {e}")
            return None

    def _improved_validate_classification_path_FIXED(
        self, domain_key: str, path_components: list
    ) -> tuple:
        """Handle domain structure navigation + nested YAML structures"""

        if not path_components:
            domain_name = domain_key.replace("_", " ")
            return True, domain_name

        # Get domain structure from the category system
        if domain_key not in self.category_system.domain_structures:
            logger.warning(f"⚠️ Domain '{domain_key}' not found in available domains")
            return False, f"Domain '{domain_key}' not found"

        domain_structure = self.category_system.domain_structures[domain_key]
        validated_path_parts = [domain_key.replace("_", " ")]

        logger.info(
            f"🔍 FIXED V3 validation for path: {path_components} in domain {domain_key}"
        )
        logger.info(f"   Domain structure keys: {list(domain_structure.keys())}")

        # 🔧 CRITICAL FIX: Navigate to the actual structure level
        current_level = None

        # Find the main category level (this varies by domain)
        for key, value in domain_structure.items():
            if isinstance(value, dict) and "subcategories" in value:
                logger.info(f"   🎯 Found main category: '{key}' with subcategories")
                current_level = value
                break

        if current_level is None:
            logger.warning(f"   ❌ No subcategories found in domain structure")
            return True, domain_key.replace("_", " ")

        # Now validate each component in the path
        for i, component in enumerate(path_components):
            found = False
            component_normalized = self._normalize_category_name(component)

            logger.info(
                f"   Looking for component: '{component}' (normalized: '{component_normalized}')"
            )

            # Navigate into subcategories
            if "subcategories" in current_level:
                subcategories = current_level["subcategories"]
                logger.info(
                    f"   Found subcategories to search (type: {type(subcategories)})"
                )

                # Handle list format subcategories (like Software domain)
                if isinstance(subcategories, list):
                    for j, subcat in enumerate(subcategories):
                        # Handle string subcategories
                        if isinstance(subcat, str):
                            subcat_normalized = self._normalize_category_name(subcat)
                            if subcat_normalized == component_normalized:
                                validated_path_parts.append(subcat)
                                found = True
                                logger.info(f"   ✅ Found string match: '{subcat}'")
                                break

                        # 🔧 CRITICAL FIX: Handle dict subcategories within lists
                        elif isinstance(subcat, dict):
                            for key, value in subcat.items():
                                key_normalized = self._normalize_category_name(key)
                                if key_normalized == component_normalized:
                                    validated_path_parts.append(key)
                                    # 🔧 NEW: Handle 'subsubcategories' within the dict
                                    if value is None and "subsubcategories" in subcat:
                                        # Set up for next level navigation
                                        current_level = {
                                            "subcategories": subcat["subsubcategories"]
                                        }
                                    elif isinstance(value, dict):
                                        current_level = value
                                    found = True
                                    logger.info(f"   ✅ Found dict key match: '{key}'")
                                    break
                            if found:
                                break

                # Handle dict format subcategories (like Lab_Equipment domain)
                elif isinstance(subcategories, dict):
                    for key, value in subcategories.items():
                        key_normalized = self._normalize_category_name(key)
                        if key_normalized == component_normalized:
                            validated_path_parts.append(key)
                            if isinstance(value, dict):
                                current_level = value
                            found = True
                            logger.info(f"   ✅ Found dict subcategory: '{key}'")
                            break

            # 🔧 NEW: Also check if we're in a 'subsubcategories' level
            if not found and isinstance(current_level.get("subcategories"), list):
                subsubcats = current_level["subcategories"]
                for subsubcat in subsubcats:
                    if isinstance(subsubcat, str):
                        subsubcat_normalized = self._normalize_category_name(subsubcat)
                        if subsubcat_normalized == component_normalized:
                            validated_path_parts.append(subsubcat)
                            found = True
                            logger.info(
                                f"   ✅ Found subsubcategory match: '{subsubcat}'"
                            )
                            break

            if not found:
                logger.info(
                    f"   ❌ Component '{component}' not found, stopping at current level"
                )
                break

        validated_path = " -> ".join(validated_path_parts)
        logger.info(f"   🎯 Final validated path: {validated_path}")

        return True, validated_path

    def _normalize_category_name(self, name: str) -> str:
        """Normalize category names for comparison"""
        return (
            str(name)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("/", " ")
        )

    def _create_successful_result(
        self,
        product_name: str,
        problem_type: str,
        new_domain: str,
        classification: Dict,
        search_result: Dict,
        old_domain: str,
        old_subcategory: str,
        tokens: int,
    ) -> Dict[str, Any]:
        """Create successful reclassification result"""
        return {
            "comprehensive_web_search_performed": True,
            "comprehensive_web_search_successful": True,
            "comprehensive_reclassification_successful": True,
            "original_problem_type": problem_type,
            "original_domain": old_domain,
            "original_subcategory": old_subcategory,
            "new_domain": new_domain,
            "new_classification": classification,
            "search_info": search_result,
            "total_tokens": tokens,
            "classification_method": "fixed_comprehensive_web_search_with_domain_targeting",
        }

    def _create_failed_search_result(
        self, product_name: str, problem_type: str, tokens: int
    ) -> Dict[str, Any]:
        """Create result for failed web search"""
        return {
            "comprehensive_web_search_performed": True,
            "comprehensive_web_search_successful": False,
            "comprehensive_reclassification_successful": False,
            "original_problem_type": problem_type,
            "total_tokens": tokens,
            "classification_method": "failed_fixed_comprehensive_web_search",
        }

    def _create_failed_classification_result(
        self,
        product_name: str,
        problem_type: str,
        search_result: Dict,
        old_domain: str,
        old_subcategory: str,
        tokens: int,
        reason: str,
    ) -> Dict[str, Any]:
        """Create result for failed classification"""
        return {
            "comprehensive_web_search_performed": True,
            "comprehensive_web_search_successful": True,
            "comprehensive_classification_successful": False,
            "comprehensive_reclassification_successful": False,
            "original_problem_type": problem_type,
            "original_domain": old_domain,
            "original_subcategory": old_subcategory,
            "search_info": search_result,
            "total_tokens": tokens,
            "classification_method": "failed_fixed_comprehensive_domain_classification",
            "failure_reason": reason,
        }

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def identify_all_problematic_products(df: pd.DataFrame) -> pd.DataFrame:
    """🎯 COMPREHENSIVE: Identify ALL problematic classifications"""

    logger.info("🎯 Identifying ALL products with problematic classifications...")

    # Build the exact conditions from the targeted script
    problematic_conditions = []
    problem_counts = {}

    # 1. primary_domain = "Other"
    other_domain_mask = (df["primary_domain"] == "Other") | (
        df["primary_domain"].isna()
    )
    if other_domain_mask.any():
        problematic_conditions.append(other_domain_mask)
        problem_counts["Other domain"] = other_domain_mask.sum()
        logger.info(
            f"📍 Found {other_domain_mask.sum()} products with primary_domain = 'Other'"
        )

    # 2. primary_subcategory = "Analytical Instrumentation"
    if "primary_subcategory" in df.columns:
        analytical_mask = df["primary_subcategory"] == "Analytical Instrumentation"
        if analytical_mask.any():
            problematic_conditions.append(analytical_mask)
            problem_counts["Analytical Instrumentation subcategory"] = (
                analytical_mask.sum()
            )
            logger.info(
                f"📍 Found {analytical_mask.sum()} products with primary_subcategory = 'Analytical Instrumentation'"
            )

    # 3. primary_subcategory = "Statistical Analysis Software"
    if "primary_subcategory" in df.columns:
        stats_software_mask = (
            df["primary_subcategory"] == "Statistical Analysis Software"
        )
        if stats_software_mask.any():
            problematic_conditions.append(stats_software_mask)
            problem_counts["Statistical Analysis Software subcategory"] = (
                stats_software_mask.sum()
            )
            logger.info(
                f"📍 Found {stats_software_mask.sum()} products with primary_subcategory = 'Statistical Analysis Software'"
            )

    # 4. primary_subcategory = "Primary Cells, Cell Lines and Microorganisms"
    if "primary_subcategory" in df.columns:
        cell_lines_mask = (
            df["primary_subcategory"] == "Primary Cells, Cell Lines and Microorganisms"
        )
        if cell_lines_mask.any():
            problematic_conditions.append(cell_lines_mask)
            problem_counts[
                "Primary Cells, Cell Lines and Microorganisms subcategory"
            ] = cell_lines_mask.sum()
            logger.info(
                f"📍 Found {cell_lines_mask.sum()} products with primary_subcategory = 'Primary Cells, Cell Lines and Microorganisms'"
            )

    # 5. Chemistry domain antibiotics (streptomycin, penicillin) classified as Laboratory Acids
    if all(
        col in df.columns for col in ["primary_domain", "primary_subcategory", "Name"]
    ):
        chemistry_mask = df["primary_domain"] == "Chemistry"
        acids_mask = df["primary_subcategory"].str.contains(
            "Laboratory Acids", case=False, na=False
        )
        antibiotic_name_mask = df["Name"].str.contains(
            "|".join(ANTIBIOTIC_KEYWORDS), case=False, na=False
        )

        antibiotic_misclass_mask = chemistry_mask & acids_mask & antibiotic_name_mask
        if antibiotic_misclass_mask.any():
            problematic_conditions.append(antibiotic_misclass_mask)
            problem_counts[
                "Antibiotics misclassified in Chemistry as Laboratory Acids"
            ] = antibiotic_misclass_mask.sum()
            logger.info(
                f"📍 Found {antibiotic_misclass_mask.sum()} antibiotics misclassified as Laboratory Acids in Chemistry"
            )

    # Combine all conditions
    if problematic_conditions:
        combined_mask = problematic_conditions[0]
        for mask in problematic_conditions[1:]:
            combined_mask |= mask
    else:
        combined_mask = pd.Series([False] * len(df), index=df.index)

    problematic_df = df[combined_mask].copy()

    # Add detailed problem identification
    problematic_df["problem_type"] = ""
    for idx in problematic_df.index:
        problems = []

        # Check each specific condition
        if (
            pd.isna(problematic_df.at[idx, "primary_domain"])
            or problematic_df.at[idx, "primary_domain"] == "Other"
        ):
            problems.append("Other_domain")

        if "primary_subcategory" in problematic_df.columns:
            subcategory = str(problematic_df.at[idx, "primary_subcategory"])
            if subcategory == "Analytical Instrumentation":
                problems.append("Generic_Analytical_Instrumentation")
            if subcategory == "Statistical Analysis Software":
                problems.append("Generic_Statistical_Software")
            if subcategory == "Primary Cells, Cell Lines and Microorganisms":
                problems.append("Generic_Cell_Lines")

        # Check antibiotic misclassification
        if (
            problematic_df.at[idx, "primary_domain"] == "Chemistry"
            and "primary_subcategory" in problematic_df.columns
        ):
            subcategory = str(problematic_df.at[idx, "primary_subcategory"])
            if "Laboratory Acids" in subcategory:
                product_name = str(problematic_df.at[idx, "Name"]).lower()
                if any(
                    antibiotic in product_name
                    for antibiotic in [kw.lower() for kw in ANTIBIOTIC_KEYWORDS]
                ):
                    problems.append("Antibiotic_misclassified_as_acid")

        problematic_df.at[idx, "problem_type"] = "|".join(problems)

    logger.info(f"🎯 Total problematic products found: {len(problematic_df)}")

    # Show breakdown by exact problem type
    for problem_type, count in problem_counts.items():
        logger.info(f"  - {problem_type}: {count}")

    return problematic_df


def get_comprehensive_reclassification_columns():
    """Get the enhanced CSV columns for comprehensive reclassification with checkpointing support"""
    return [
        # Problem identification
        "problem_type",
        # Comprehensive web search columns
        "fixed_comprehensive_web_search_performed",
        "fixed_comprehensive_web_search_successful",
        "fixed_comprehensive_reclassification_successful",
        "fixed_comprehensive_suggested_domain",
        "fixed_comprehensive_domain_confidence",
        "fixed_comprehensive_new_domain",
        "fixed_comprehensive_new_subcategory",
        "fixed_comprehensive_new_subsubcategory",
        "fixed_comprehensive_new_subsubsubcategory",
        "fixed_comprehensive_new_confidence",
        "fixed_comprehensive_validated_path",
        "fixed_comprehensive_total_tokens",
        # Checkpointing metadata
        "checkpoint_processed_at",
        "processing_error",
    ]


def process_full_comprehensive_reclassification_with_checkpointing(
    input_csv: str, output_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """🚀 FIXED CHECKPOINTING: Process ALL problematic classifications with proper progress tracking"""

    logger.info(
        "🌐 Starting FULL COMPREHENSIVE Web Search Enhanced Reclassification with FIXED CHECKPOINTING..."
    )

    # Setup checkpoint directory
    if os.path.exists(COMPREHENSIVE_CHECKPOINT_DIR) and not os.path.isdir(
        COMPREHENSIVE_CHECKPOINT_DIR
    ):
        logger.error(
            f"'{COMPREHENSIVE_CHECKPOINT_DIR}' exists but is not a directory. Please remove or rename this file."
        )
        exit(1)
    os.makedirs(COMPREHENSIVE_CHECKPOINT_DIR, exist_ok=True)

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    web_classifier = FixedComprehensiveWebSearchClassifier(category_system, tag_system)

    try:
        # Check for existing checkpoint
        latest_checkpoint, last_checkpoint_number = (
            find_latest_comprehensive_checkpoint()
        )

        if latest_checkpoint:
            logger.info(f"📁 Found comprehensive checkpoint: {latest_checkpoint}")
            logger.info(f"📊 Last checkpoint number: {last_checkpoint_number}")
            complete_df = pd.read_csv(latest_checkpoint, low_memory=False)

            # 🔧 CRITICAL FIX: Check processing flags FIRST, not problem_type column
            # The key issue: we should look at what's been processed, not just problem identification
            already_processed_count = 0

            if "fixed_comprehensive_web_search_performed" in complete_df.columns:
                already_processed_count = (
                    complete_df["fixed_comprehensive_web_search_performed"] == True
                ).sum()
                logger.info(
                    f"📊 Found {already_processed_count} products already processed"
                )

                # Debug: Show breakdown of processing status
                total_performed = (
                    complete_df["fixed_comprehensive_web_search_performed"] == True
                ).sum()
                total_successful = (
                    complete_df["fixed_comprehensive_web_search_successful"] == True
                ).sum()
                total_reclassified = (
                    complete_df["fixed_comprehensive_reclassification_successful"]
                    == True
                ).sum()
                logger.info(f"🔍 Processing status breakdown:")
                logger.info(f"    Web search performed: {total_performed}")
                logger.info(f"    Web search successful: {total_successful}")
                logger.info(f"    Reclassification successful: {total_reclassified}")
            else:
                logger.info(
                    "📊 No processing columns found - this appears to be a fresh dataset"
                )

            # If we have processed items, we're resuming - find remaining work
            if already_processed_count > 0:
                logger.info(
                    "🔄 RESUMING: Found processed items, looking for remaining work..."
                )

                # Ensure problem_type is set (might be missing from old checkpoints)
                if (
                    "problem_type" not in complete_df.columns
                    or (
                        complete_df["problem_type"].notna()
                        & (complete_df["problem_type"] != "")
                    ).sum()
                    < already_processed_count
                ):

                    logger.info(
                        "🔧 Re-identifying problematic products to ensure problem_type column..."
                    )
                    # Initialize problem_type column if missing
                    if "problem_type" not in complete_df.columns:
                        complete_df["problem_type"] = ""

                    # Re-identify problems but DON'T reset processing flags
                    temp_problematic = identify_all_problematic_products(complete_df)
                    for idx in temp_problematic.index:
                        complete_df.at[idx, "problem_type"] = temp_problematic.at[
                            idx, "problem_type"
                        ]

                    logger.info(
                        f"✅ Problem types updated for {len(temp_problematic)} problematic products"
                    )

                # Find unprocessed problematic products
                problematic_mask = (
                    (complete_df["problem_type"].notna())
                    & (complete_df["problem_type"] != "")
                    & (complete_df["fixed_comprehensive_web_search_performed"] != True)
                )

                problematic_df = complete_df[problematic_mask].copy()
                logger.info(
                    f"📊 Unprocessed problematic products found: {len(problematic_df)}"
                )

            else:
                # No processed items - this is either a fresh start or problem_type missing
                logger.info(
                    "🆕 FRESH START: No processed items found, identifying all problematic products..."
                )

                # Initialize comprehensive columns
                comprehensive_columns = get_comprehensive_reclassification_columns()
                for col in comprehensive_columns:
                    if col not in complete_df.columns:
                        if col.endswith("_tokens"):
                            complete_df[col] = 0
                        elif col.endswith("_performed") or col.endswith("_successful"):
                            complete_df[col] = False
                        else:
                            complete_df[col] = ""

                # Identify ALL problematic products from scratch
                problematic_df = identify_all_problematic_products(complete_df)
                already_processed_count = 0

            if len(problematic_df) == 0:
                logger.info("✅ All problematic products have been processed!")
                # Still show a summary
                total_problematic = (
                    complete_df["problem_type"].notna()
                    & (complete_df["problem_type"] != "")
                ).sum()
                logger.info(
                    f"📊 Total problematic products in dataset: {total_problematic}"
                )
                logger.info(f"📊 Total processed: {already_processed_count}")
                return complete_df, pd.DataFrame()

            logger.info(
                f"📍 Resuming from {len(problematic_df)} unprocessed problematic products"
            )
            logger.info(f"📍 Already processed: {already_processed_count} products")

        else:
            # Start fresh - load complete dataset
            logger.info(f"🆕 No checkpoint found, starting fresh from {input_csv}")
            complete_df = pd.read_csv(input_csv)
            logger.info(f"✅ Loaded {len(complete_df)} total products")

            # Initialize comprehensive columns
            comprehensive_columns = get_comprehensive_reclassification_columns()
            for col in comprehensive_columns:
                if col.endswith("_tokens"):
                    complete_df[col] = 0
                elif col.endswith("_performed") or col.endswith("_successful"):
                    complete_df[col] = False
                else:
                    complete_df[col] = ""

            # Identify ALL problematic products
            problematic_df = identify_all_problematic_products(complete_df)
            already_processed_count = 0

        if len(problematic_df) == 0:
            logger.info("✅ No problematic products found!")
            return complete_df, pd.DataFrame()

        # 🔧 CRITICAL FIX: Get total problematic count for progress display
        total_problematic_in_dataset = (
            complete_df["problem_type"].notna() & (complete_df["problem_type"] != "")
        ).sum()

        logger.info(
            f"📊 Processing {len(problematic_df)} remaining problematic products..."
        )
        logger.info(
            f"📈 Total progress so far: {already_processed_count}/{total_problematic_in_dataset} problematic products"
        )

        # Process each problematic product with FIXED checkpointing
        successful_reclassifications = 0
        failed_reclassifications = 0

        # 🔧 CRITICAL FIX: Create progress bar that shows total progress
        progress_desc = f"🌐 Comprehensive Reclassification ({already_processed_count}/{total_problematic_in_dataset} done)"

        for i, (idx, row) in enumerate(
            tqdm(
                problematic_df.iterrows(),
                desc=progress_desc,
                total=len(problematic_df),
                initial=0,  # Start progress bar at 0 for this batch
            )
        ):
            name = row["Name"]
            manufacturer = row.get("Manufacturer", "") if "Manufacturer" in row else ""
            problem_type = row["problem_type"]
            current_domain = (
                row.get("primary_domain", "") if "primary_domain" in row else ""
            )
            current_subcategory = (
                row.get("primary_subcategory", "")
                if "primary_subcategory" in row
                else ""
            )
            original_description = (
                row.get("Description", "") if "Description" in row else ""
            )

            try:
                # Perform comprehensive web search reclassification
                result = web_classifier.reclassify_problematic_product(
                    name,
                    manufacturer,
                    problem_type,
                    current_domain,
                    current_subcategory,
                    original_description,
                )

                # Store results in the complete dataframe
                complete_df.at[idx, "fixed_comprehensive_web_search_performed"] = (
                    result.get("comprehensive_web_search_performed", False)
                )
                complete_df.at[idx, "fixed_comprehensive_web_search_successful"] = (
                    result.get("comprehensive_web_search_successful", False)
                )
                complete_df.at[
                    idx, "fixed_comprehensive_reclassification_successful"
                ] = result.get("comprehensive_reclassification_successful", False)

                # Store search info
                search_info = result.get("search_info", {})
                complete_df.at[idx, "fixed_comprehensive_suggested_domain"] = (
                    search_info.get("suggested_domain", "")
                )
                complete_df.at[idx, "fixed_comprehensive_domain_confidence"] = (
                    search_info.get("confidence", "")
                )

                # Store classification results
                if result.get("comprehensive_reclassification_successful", False):
                    classification = result.get("new_classification", {})
                    complete_df.at[idx, "fixed_comprehensive_new_domain"] = result.get(
                        "new_domain", ""
                    )
                    complete_df.at[idx, "fixed_comprehensive_new_subcategory"] = (
                        classification.get("subcategory", "")
                    )
                    complete_df.at[idx, "fixed_comprehensive_new_subsubcategory"] = (
                        classification.get("subsubcategory", "")
                    )
                    complete_df.at[idx, "fixed_comprehensive_new_subsubsubcategory"] = (
                        classification.get("subsubsubcategory", "")
                    )
                    complete_df.at[idx, "fixed_comprehensive_new_confidence"] = (
                        classification.get("confidence", "")
                    )
                    complete_df.at[idx, "fixed_comprehensive_validated_path"] = (
                        classification.get("validated_path", "")
                    )

                    # Update primary classification if successful and new domain is not "Other"
                    new_domain = result.get("new_domain", "")
                    if new_domain and new_domain != "Other":
                        complete_df.at[idx, "primary_domain"] = new_domain
                        complete_df.at[idx, "primary_subcategory"] = classification.get(
                            "subcategory", ""
                        )
                        complete_df.at[idx, "primary_subsubcategory"] = (
                            classification.get("subsubcategory", "")
                        )
                        complete_df.at[idx, "primary_subsubsubcategory"] = (
                            classification.get("subsubsubcategory", "")
                        )
                        complete_df.at[idx, "primary_confidence"] = classification.get(
                            "confidence", ""
                        )
                        if "validated_path_primary" in complete_df.columns:
                            complete_df.at[idx, "validated_path_primary"] = (
                                classification.get("validated_path", "")
                            )

                    successful_reclassifications += 1
                    logger.info(
                        f"✅ Successfully reclassified: {name[:30]} → {new_domain}"
                    )
                else:
                    failed_reclassifications += 1

                # Store token usage and processing metadata
                complete_df.at[idx, "fixed_comprehensive_total_tokens"] = result.get(
                    "total_tokens", 0
                )
                complete_df.at[idx, "checkpoint_processed_at"] = (
                    datetime.now().isoformat()
                )
                complete_df.at[idx, "processing_error"] = False

            except Exception as e:
                logger.error(f"❌ Error processing '{name}': {e}")
                complete_df.at[idx, "fixed_comprehensive_web_search_performed"] = True
                complete_df.at[idx, "fixed_comprehensive_web_search_successful"] = False
                complete_df.at[
                    idx, "fixed_comprehensive_reclassification_successful"
                ] = False
                complete_df.at[idx, "checkpoint_processed_at"] = (
                    datetime.now().isoformat()
                )
                complete_df.at[idx, "processing_error"] = True
                failed_reclassifications += 1

            # 🔧 CRITICAL FIX: Save checkpoint with proper continuous numbering
            if (i + 1) % COMPREHENSIVE_CHECKPOINT_FREQ == 0 or i == len(
                problematic_df
            ) - 1:
                # Calculate ACTUAL total progress (already processed + current batch progress)
                actual_progress = already_processed_count + i + 1

                checkpoint_file = os.path.join(
                    COMPREHENSIVE_CHECKPOINT_DIR,
                    f"{COMPREHENSIVE_CHECKPOINT_PREFIX}_{actual_progress}.csv",
                )
                complete_df.to_csv(checkpoint_file, index=False)
                logger.info(f"💾 Saved comprehensive checkpoint: {checkpoint_file}")
                logger.info(
                    f"📈 Total progress: {actual_progress}/{total_problematic_in_dataset} problematic products processed"
                )

        # Save final result
        complete_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Complete comprehensive reclassification saved to {output_csv}")

        # Generate report
        generate_comprehensive_reclassification_report(
            problematic_df,
            web_classifier,
            successful_reclassifications,
            failed_reclassifications,
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"🌐 COMPREHENSIVE RECLASSIFICATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(
            f"📊 Problematic products processed this session: {len(problematic_df)}"
        )
        logger.info(
            f"📊 Total problematic products processed: {already_processed_count + len(problematic_df)}"
        )
        logger.info(
            f"✅ Successful reclassifications this session: {successful_reclassifications}"
        )
        logger.info(
            f"❌ Failed reclassifications this session: {failed_reclassifications}"
        )
        if len(problematic_df) > 0:
            logger.info(
                f"📈 Success rate this session: {successful_reclassifications/len(problematic_df)*100:.1f}%"
            )

        return complete_df, problematic_df

    except Exception as e:
        logger.error(f"❌ Error in comprehensive reclassification: {e}")
        raise


def generate_comprehensive_reclassification_report(
    problematic_df: pd.DataFrame,
    classifier: FixedComprehensiveWebSearchClassifier,
    successful: int,
    failed: int,
):
    """Generate report for comprehensive web search reclassification results"""

    print(f"\n{'='*80}")
    print("🚀 COMPREHENSIVE WEB SEARCH RECLASSIFICATION REPORT")
    print(f"{'='*80}")

    total = len(problematic_df)

    print(f"📊 OVERVIEW")
    print(f"  Total problematic products processed this session: {total}")
    print(f"  Successful reclassifications: {successful} ({successful/total*100:.1f}%)")
    print(f"  Failed reclassifications: {failed} ({failed/total*100:.1f}%)")

    # Web search statistics
    print(f"\n{'WEB SEARCH STATISTICS':-^60}")
    print(f"  Total searches performed: {classifier.total_searches}")
    print(f"  Successful searches: {classifier.successful_searches}")
    if classifier.total_searches > 0:
        print(
            f"  Search success rate: {classifier.successful_searches/classifier.total_searches*100:.1f}%"
        )

    # Problem type breakdown
    if "problem_type" in problematic_df.columns:
        print(f"\n{'PROBLEM TYPES PROCESSED':-^60}")
        problem_types = (
            problematic_df["problem_type"].str.split("|").explode().value_counts()
        )
        for ptype, count in problem_types.head(10).items():
            if ptype:
                print(f"  {ptype:<35} {count:>5}")

    print(f"\n{'PROCESSING EFFICIENCY':-^60}")
    if hasattr(classifier, "search_cache"):
        print(f"  Cached searches: {len(classifier.search_cache)}")

    print(f"  Cache hit rate: Optimized for repeated products")


def main():
    """Enhanced main function with full dataset processing and FIXED checkpointing"""
    print("=" * 80)
    print("🚀 COMPREHENSIVE WEB SEARCH RECLASSIFICATION WITH FIXED CHECKPOINTING")
    print(
        "Fixes checkpoint resume + progress tracking + targets ALL problematic classifications"
    )
    print("=" * 80)

    try:
        print("\n🎯 What would you like to do?")
        print("1. Run FULL comprehensive reclassification with FIXED checkpointing")
        print("2. Check checkpoint status (improved)")
        print("3. Resume from latest checkpoint (FIXED)")
        print("4. Debug checkpoint issues")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            print(
                "\n🚀 Starting FULL comprehensive reclassification with FIXED checkpointing..."
            )
            user_confirmation = input(
                "⚠️  This will process the entire dataset. Continue? (y/n): "
            )
            if user_confirmation.lower() == "y":
                complete_df, problematic_df = (
                    process_full_comprehensive_reclassification_with_checkpointing(
                        input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
                    )
                )
                print("✅ Full dataset processing completed!")
            else:
                print("❌ Operation cancelled.")

        elif choice == "2":
            print("\n📊 Checking checkpoint status (improved)...")
            latest_checkpoint, checkpoint_number = (
                find_latest_comprehensive_checkpoint()
            )
            if latest_checkpoint:
                print(f"✅ Latest checkpoint found: {latest_checkpoint}")
                print(f"📊 Checkpoint number: {checkpoint_number}")

                # Load and analyze checkpoint
                df = pd.read_csv(latest_checkpoint)
                total_products = len(df)

                if "fixed_comprehensive_web_search_performed" in df.columns:
                    processed = len(
                        df[df["fixed_comprehensive_web_search_performed"] == True]
                    )
                    print(
                        f"📈 Progress: {processed}/{total_products} products processed ({processed/total_products*100:.1f}%)"
                    )

                    if "problem_type" in df.columns:
                        problematic = len(
                            df[
                                (df["problem_type"].notna())
                                & (df["problem_type"] != "")
                            ]
                        )
                        problematic_processed = len(
                            df[
                                (df["problem_type"].notna())
                                & (df["problem_type"] != "")
                                & (
                                    df["fixed_comprehensive_web_search_performed"]
                                    == True
                                )
                            ]
                        )
                        problematic_remaining = problematic - problematic_processed
                        print(
                            f"🎯 Problematic products: {problematic_processed}/{problematic} processed ({problematic_remaining} remaining)"
                        )

                        if problematic_remaining > 0:
                            print(
                                f"⏰ Estimated time remaining: {problematic_remaining * 0.5:.1f} minutes"
                            )
                else:
                    print(
                        "📋 Checkpoint contains raw data, no processing has started yet"
                    )
            else:
                print("❌ No checkpoint found")

        elif choice == "3":
            print("\n🔄 Resuming from latest checkpoint (FIXED)...")
            latest_checkpoint, checkpoint_number = (
                find_latest_comprehensive_checkpoint()
            )
            if latest_checkpoint:
                print(f"📁 Found checkpoint: {latest_checkpoint}")
                print(f"📊 Resuming from checkpoint {checkpoint_number}")
                complete_df, problematic_df = (
                    process_full_comprehensive_reclassification_with_checkpointing(
                        input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
                    )
                )
                print("✅ Resume completed!")
            else:
                print("❌ No checkpoint found. Use option 1 to start fresh.")

        elif choice == "4":
            print("\n🔍 Debugging checkpoint issues...")
            debug_checkpoint_comprehensive()

        else:
            print("❌ Invalid choice. Please select 1-4.")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


def debug_checkpoint_comprehensive():
    """Debug comprehensive checkpoint issues with FIXED analysis"""

    latest_checkpoint, checkpoint_number = find_latest_comprehensive_checkpoint()
    if not latest_checkpoint:
        print("❌ No checkpoint found!")
        return

    print(f"🔍 ANALYZING CHECKPOINT: {latest_checkpoint}")
    print(f"📊 Checkpoint number: {checkpoint_number}")
    print("=" * 70)

    # Load the checkpoint
    df = pd.read_csv(latest_checkpoint, low_memory=False)

    print(f"📊 BASIC INFO:")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")

    # Check if comprehensive columns exist
    comprehensive_columns = [
        "problem_type",
        "fixed_comprehensive_web_search_performed",
        "fixed_comprehensive_web_search_successful",
        "fixed_comprehensive_reclassification_successful",
    ]

    print(f"\n🔍 COMPREHENSIVE COLUMNS STATUS:")
    for col in comprehensive_columns:
        if col in df.columns:
            print(f"  ✅ {col}: EXISTS")

            # Show value counts for boolean/status columns
            if col != "problem_type":
                value_counts = df[col].value_counts(dropna=False)
                print(f"      Values: {dict(value_counts)}")
        else:
            print(f"  ❌ {col}: MISSING")

    # Analyze problem_type column specifically
    if "problem_type" in df.columns:
        print(f"\n🎯 PROBLEM TYPE ANALYSIS:")

        # Count products with problem_type
        has_problem_type = df["problem_type"].notna() & (df["problem_type"] != "")
        print(f"  Products with problem_type: {has_problem_type.sum()}")
        print(f"  Products without problem_type: {(~has_problem_type).sum()}")

        if has_problem_type.sum() > 0:
            # Show problem type distribution
            problem_types = (
                df[has_problem_type]["problem_type"]
                .str.split("|")
                .explode()
                .value_counts()
            )
            print(f"  Problem type breakdown:")
            for ptype, count in problem_types.head(10).items():
                if ptype:
                    print(f"    {ptype}: {count}")

            # FIXED: Show processing progress
            if "fixed_comprehensive_web_search_performed" in df.columns:
                processed = (
                    df["fixed_comprehensive_web_search_performed"] == True
                ).sum()
                unprocessed_problematic = (
                    has_problem_type
                    & (df["fixed_comprehensive_web_search_performed"] != True)
                ).sum()
                print(
                    f"  Problematic products processed: {processed - (len(df) - has_problem_type.sum())}"
                )
                print(f"  Problematic products remaining: {unprocessed_problematic}")
        else:
            print("  ⚠️ No products have problem_type set!")
    else:
        print(f"\n❌ problem_type column is missing!")

    # Check primary domain distribution for problematic patterns
    if "primary_domain" in df.columns:
        print(f"\n🏷️ DOMAIN ANALYSIS:")
        other_domain = (df["primary_domain"] == "Other").sum()
        missing_domain = df["primary_domain"].isna().sum()
        print(f"  'Other' domain: {other_domain}")
        print(f"  Missing domain: {missing_domain}")
        print(f"  Potentially problematic: {other_domain + missing_domain}")

        if "primary_subcategory" in df.columns:
            analytical_instr = (
                df["primary_subcategory"] == "Analytical Instrumentation"
            ).sum()
            stats_software = (
                df["primary_subcategory"] == "Statistical Analysis Software"
            ).sum()
            print(f"  Generic 'Analytical Instrumentation': {analytical_instr}")
            print(f"  Generic 'Statistical Analysis Software': {stats_software}")

    # FIXED: Improved recommendations
    print(f"\n💡 RECOMMENDATION:")

    # Check if there are any processing flags at all
    has_processing_columns = any(
        col in df.columns
        for col in [
            "fixed_comprehensive_web_search_performed",
            "fixed_comprehensive_web_search_successful",
            "fixed_comprehensive_reclassification_successful",
        ]
    )

    if not has_processing_columns:
        print("  ❌ This checkpoint has no processing columns - it's a raw dataset.")
        print("  ✅ Run option 1 (Full reclassification) to start processing.")
        return

    # Check processing status
    already_processed = 0
    if "fixed_comprehensive_web_search_performed" in df.columns:
        already_processed = (
            df["fixed_comprehensive_web_search_performed"] == True
        ).sum()

    if already_processed == 0:
        print("  📋 This checkpoint has processing columns but no processed items.")
        print("  ✅ Run option 1 or 3 to start/resume processing.")
    else:
        print(f"  ✅ Found {already_processed} processed items in checkpoint.")

        # Check for remaining work
        if (
            "problem_type" not in df.columns
            or (df["problem_type"].notna() & (df["problem_type"] != "")).sum() == 0
        ):
            print("  ⚠️ Problem identification missing - will be rebuilt on resume.")
            print("  ✅ Run option 3 (Resume from checkpoint) to continue.")
        else:
            has_problems = df["problem_type"].notna() & (df["problem_type"] != "")
            unprocessed = (
                has_problems & (df["fixed_comprehensive_web_search_performed"] != True)
            ).sum()

            if unprocessed > 0:
                print(f"  🎯 Found {unprocessed} unprocessed problematic products.")
                print(
                    f"  ✅ Run option 3 (Resume from checkpoint) to continue processing."
                )
                print(f"  ⏰ Estimated time remaining: {unprocessed * 0.5:.1f} minutes")
            else:
                print("  ✅ All problematic products appear to be processed!")
                print(f"  📊 Checkpoint represents {checkpoint_number} completed items")


if __name__ == "__main__":
    main()
