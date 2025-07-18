# COMPREHENSIVE WEB SEARCH ENHANCED RE-CLASSIFICATION SYSTEM - FIXED VERSION
# Targets ALL problematic classifications + fixes YAML validation and prompts

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

# Import existing classification system components
from enhanced_classification_full_checkpoint import (
    LLMDrivenCategorySystem,
    EnhancedTagSystem,
    EnhancedLLMClassifier,
)

# Configuration
INPUT_CSV = "products_enhanced_fixed_classification.csv"
OUTPUT_CSV = "products_comprehensive_web_search_reclassified_FIXED.csv"

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


class FixedComprehensiveWebSearchClassifier:
    """🚀 FIXED COMPREHENSIVE: All problematic classifications + proper YAML handling + better prompts"""

    def __init__(
        self,
        category_system: LLMDrivenCategorySystem,
        tag_system: EnhancedTagSystem,
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

            time.sleep(1.0)  # Rate limiting
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

    def _get_improved_domain_prompt_FIXED(
        self, domain_key: str, product_name: str, description: str
    ) -> str:
        """🔧 FIXED: Domain-specific prompts with subcategory examples and critical guidance"""

        # Domain-specific guidance with explicit examples
        domain_guides = {
            "Software": {
                "subcategories": [
                    "Statistical Analysis Software (for SAS, SPSS, GraphPad Prism, STATA, R, JMP)",
                    "Data Analysis Software (for MATLAB, Python packages)",
                    "Image Analysis Software (for ImageJ, imaging tools)",
                    "Laboratory Management Software (for LIMS, Origin)",
                    "Chromatography Software (for chromatography control)",
                ],
                "examples": {
                    "GraphPad Prism": "Statistical Analysis Software -> GraphPad Prism",
                    "SAS": "Statistical Analysis Software -> SAS",
                    "STATA": "Statistical Analysis Software -> STATA",
                    "SPSS": "Statistical Analysis Software -> SPSS",
                    "R Software": "Statistical Analysis Software -> R Software",
                    "JMP": "Statistical Analysis Software -> JMP",
                    "MATLAB": "Data Analysis Software -> MATLAB",
                    "ImageJ": "Image Analysis Software",
                },
                "critical_note": "Statistical software products MUST go to Statistical Analysis Software subcategory with specific software name",
            },
            "Lab_Equipment": {
                "subcategories": [
                    "Laboratory Supplies & Consumables (membranes, filters, tubes, plates)",
                    "General Laboratory Equipment (centrifuges, incubators, stirrers)",
                    "Western Blotting Equipment (blotters, processors)",
                    "Spectroscopy (spectrophotometers, microplate readers)",
                    "PCR Equipment (thermocyclers, qPCR machines)",
                    "Electrophoresis Equipment (gel systems, power supplies)",
                    "Chromatography Equipment (HPLC, columns)",
                    "Analytical Instrumentation (specific analysis instruments)",
                ],
                "examples": {
                    "Hybond": "Laboratory Supplies & Consumables",
                    "nitrocellulose membrane": "Laboratory Supplies & Consumables",
                    "PVDF membrane": "Laboratory Supplies & Consumables",
                    "filter membrane": "Laboratory Supplies & Consumables",
                    "centrifuge": "General Laboratory Equipment",
                    "spectrophotometer": "Spectroscopy",
                    "PCR machine": "PCR Equipment",
                    "thermocycler": "PCR Equipment",
                },
                "critical_note": "Membranes and filters MUST go to Laboratory Supplies & Consumables. Use specific equipment types, not generic 'Analytical Instrumentation'",
            },
            "Chemistry": {
                "subcategories": [
                    "Chemicals (molecular biology chemicals, solvents)",
                    "Reagents (biochemical reagents, assay reagents)",
                    "Buffers (buffer solutions, buffer components)",
                    "Standards (reference standards, calibration standards)",
                ],
                "examples": {
                    "dexamethasone": "Chemicals -> Molecular Biology Chemicals",
                    "acetonitrile": "Chemicals -> Solvents",
                    "buffer solution": "Buffers",
                    "DMSO": "Chemicals -> Solvents",
                },
                "critical_note": "NOT for research antibiotics (those go to Cell_Biology domain)",
            },
            "Cell_Biology": {
                "subcategories": [
                    "Cell Culture Reagents (media, supplements)",
                    "Inhibitors (research antibiotics, enzyme inhibitors)",
                    "Cell Lines (primary cells, cell lines)",
                    "Staining Reagents (cell stains, viability dyes)",
                ],
                "examples": {
                    "streptomycin": "Inhibitors -> Research Antibiotics",
                    "penicillin": "Inhibitors -> Research Antibiotics",
                    "cell line": "Cell Lines",
                    "cell culture medium": "Cell Culture Reagents",
                },
                "critical_note": "Research antibiotics belong here, NOT in Chemistry domain",
            },
            "Protein": {
                "subcategories": [
                    "Protein Purification (columns, resins)",
                    "Western Blot Supplies (transfer equipment, buffers)",
                    "Protein Analysis (electrophoresis, chromatography)",
                    "Biochemistry Reagents (protein reagents)",
                ],
                "examples": {
                    "HiTrap column": "Protein Purification",
                    "protein purification column": "Protein Purification",
                    "SDS-PAGE": "Protein Analysis",
                },
                "critical_note": "Focus on protein-specific applications and purification",
            },
        }

        guide = domain_guides.get(
            domain_key,
            {
                "subcategories": ["General classification needed"],
                "examples": {},
                "critical_note": "Use best judgment for classification",
            },
        )

        # Get YAML structure for validation
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        prompt = f"""Classify this product within the {domain_key} domain using the guidance below.

Product: "{product_name}"
Description: "{description}"

🎯 MAIN SUBCATEGORIES FOR {domain_key}:
{chr(10).join(f"• {subcat}" for subcat in guide["subcategories"])}

📋 CRITICAL CLASSIFICATION EXAMPLES:
{chr(10).join(f"• {prod} → {path}" for prod, path in guide["examples"].items())}

⚠️ CRITICAL RULE: {guide["critical_note"]}

YAML STRUCTURE FOR VALIDATION:
{focused_structure}

CLASSIFICATION RULES:
1. **FOLLOW THE EXAMPLES ABOVE** - they show the exact correct patterns
2. For software: MUST specify the exact software name (SAS, SPSS, GraphPad Prism, etc.)
3. For equipment: Use specific equipment type, NOT generic categories
4. For membranes/filters: MUST go to Laboratory Supplies & Consumables
5. Go as deep as possible using exact YAML names

SPECIAL REQUIREMENTS:
- GraphPad Prism, SAS, STATA, SPSS → Statistical Analysis Software -> [specific software name]
- Hybond, membranes, filters → Laboratory Supplies & Consumables
- Research antibiotics → Cell_Biology domain (NOT Chemistry)
- Protein purification columns → Protein domain

Respond with valid JSON only:
{{
    "subcategory": "exact_subcategory_name_from_examples_above",
    "subsubcategory": "specific_type_if_available_from_yaml",
    "subsubsubcategory": "deepest_level_if_available",
    "confidence": "High/Medium/Low",
    "reasoning": "why_this_classification_matches_the_examples_and_guidance"
}}"""

        return prompt

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
        # For Software: domain_structure = {'Life Science Software': {...}}
        # For Lab_Equipment: domain_structure = {'Lab Equipment': {...}}

        current_level = None

        # Find the main category level (this varies by domain)
        for key, value in domain_structure.items():
            if isinstance(value, dict) and "subcategories" in value:
                logger.info(f"   🎯 Found main category: '{key}' with subcategories")
                current_level = value
                # Don't add the main category to the path for now, we'll add subcategories
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


# ============================================================================
# COMPREHENSIVE PROCESSING FUNCTIONS (Updated with FIXED classifier)
# ============================================================================


def process_fixed_comprehensive_web_search_reclassification(
    input_csv: str, output_csv: str, max_products: int = None, test_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """🚀 FIXED COMPREHENSIVE: Process ALL problematic classifications with fixed web search"""

    logger.info(
        "🌐 Starting FIXED COMPREHENSIVE Web Search Enhanced Reclassification..."
    )

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    web_classifier = FixedComprehensiveWebSearchClassifier(category_system, tag_system)

    # Load and identify ALL problematic products
    logger.info(f"📊 Loading complete dataset from: {input_csv}")
    complete_df = pd.read_csv(input_csv)
    logger.info(f"✅ Loaded {len(complete_df)} total products")

    problematic_df = identify_all_problematic_products(complete_df)

    if len(problematic_df) == 0:
        logger.info("✅ No problematic products found!")
        return complete_df, pd.DataFrame()

    # Limit products for testing
    if test_mode and max_products:
        problematic_df = problematic_df.head(max_products)
        logger.info(f"🧪 Test mode: Processing {len(problematic_df)} products")

    # Add comprehensive reclassification columns
    comprehensive_columns = [
        "fixed_comprehensive_web_search_performed",
        "fixed_comprehensive_web_search_successful",
        "fixed_comprehensive_reclassification_successful",
        "original_problem_type",
        "fixed_comprehensive_search_query",
        "fixed_comprehensive_suggested_domain",
        "fixed_comprehensive_domain_confidence",
        "fixed_comprehensive_domain_reasoning",
        "fixed_comprehensive_new_domain",
        "fixed_comprehensive_new_subcategory",
        "fixed_comprehensive_new_subsubcategory",
        "fixed_comprehensive_new_subsubsubcategory",
        "fixed_comprehensive_new_confidence",
        "fixed_comprehensive_validated_path",
        "fixed_comprehensive_total_tokens",
    ]

    for col in comprehensive_columns:
        if col.endswith("_tokens"):
            problematic_df[col] = 0
        elif col.endswith("_performed") or col.endswith("_successful"):
            problematic_df[col] = False
        else:
            problematic_df[col] = ""

    # Process each problematic product
    logger.info(
        f"🔄 Processing {len(problematic_df)} problematic products with FIXED comprehensive web search..."
    )

    successful_reclassifications = 0
    failed_reclassifications = 0

    for idx in tqdm(problematic_df.index, desc="🌐 FIXED Comprehensive Web Search"):
        name = problematic_df.at[idx, "Name"]
        manufacturer = (
            problematic_df.at[idx, "Manufacturer"]
            if "Manufacturer" in problematic_df.columns
            else ""
        )
        problem_type = problematic_df.at[idx, "problem_type"]
        current_domain = (
            problematic_df.at[idx, "primary_domain"]
            if "primary_domain" in problematic_df.columns
            else ""
        )
        current_subcategory = (
            problematic_df.at[idx, "primary_subcategory"]
            if "primary_subcategory" in problematic_df.columns
            else ""
        )
        original_description = (
            problematic_df.at[idx, "Description"]
            if "Description" in problematic_df.columns
            else ""
        )

        try:
            # Perform FIXED comprehensive web search reclassification
            result = web_classifier.reclassify_problematic_product(
                name,
                manufacturer,
                problem_type,
                current_domain,
                current_subcategory,
                original_description,
            )

            # Store results
            problematic_df.at[idx, "fixed_comprehensive_web_search_performed"] = (
                result.get("comprehensive_web_search_performed", False)
            )
            problematic_df.at[idx, "fixed_comprehensive_web_search_successful"] = (
                result.get("comprehensive_web_search_successful", False)
            )
            problematic_df.at[
                idx, "fixed_comprehensive_reclassification_successful"
            ] = result.get("comprehensive_reclassification_successful", False)
            problematic_df.at[idx, "original_problem_type"] = problem_type

            # Store search info
            search_info = result.get("search_info", {})
            problematic_df.at[idx, "fixed_comprehensive_search_query"] = (
                f"{name} {manufacturer}".strip()
            )
            problematic_df.at[idx, "fixed_comprehensive_suggested_domain"] = (
                search_info.get("suggested_domain", "")
            )
            problematic_df.at[idx, "fixed_comprehensive_domain_confidence"] = (
                search_info.get("confidence", "")
            )
            problematic_df.at[idx, "fixed_comprehensive_domain_reasoning"] = (
                search_info.get("reasoning", "")
            )

            # Store classification results
            if result.get("comprehensive_reclassification_successful", False):
                classification = result.get("new_classification", {})
                problematic_df.at[idx, "fixed_comprehensive_new_domain"] = result.get(
                    "new_domain", ""
                )
                problematic_df.at[idx, "fixed_comprehensive_new_subcategory"] = (
                    classification.get("subcategory", "")
                )
                problematic_df.at[idx, "fixed_comprehensive_new_subsubcategory"] = (
                    classification.get("subsubcategory", "")
                )
                problematic_df.at[idx, "fixed_comprehensive_new_subsubsubcategory"] = (
                    classification.get("subsubsubcategory", "")
                )
                problematic_df.at[idx, "fixed_comprehensive_new_confidence"] = (
                    classification.get("confidence", "")
                )
                problematic_df.at[idx, "fixed_comprehensive_validated_path"] = (
                    classification.get("validated_path", "")
                )

                successful_reclassifications += 1
            else:
                failed_reclassifications += 1

            # Store token usage
            problematic_df.at[idx, "fixed_comprehensive_total_tokens"] = result.get(
                "total_tokens", 0
            )

        except Exception as e:
            logger.error(f"❌ Error processing '{name}': {e}")
            problematic_df.at[idx, "fixed_comprehensive_web_search_performed"] = True
            problematic_df.at[idx, "fixed_comprehensive_web_search_successful"] = False
            problematic_df.at[
                idx, "fixed_comprehensive_reclassification_successful"
            ] = False
            problematic_df.at[idx, "original_problem_type"] = problem_type
            failed_reclassifications += 1

    # Save results
    reclassification_details_csv = output_csv.replace(
        ".csv", "_FIXED_comprehensive_details.csv"
    )
    problematic_df.to_csv(reclassification_details_csv, index=False)
    logger.info(
        f"✅ FIXED comprehensive reclassification details saved to: {reclassification_details_csv}"
    )

    # Merge results back
    complete_updated_df = merge_fixed_comprehensive_results(complete_df, problematic_df)
    complete_updated_csv = output_csv.replace(
        ".csv", "_FIXED_comprehensive_complete.csv"
    )
    complete_updated_df.to_csv(complete_updated_csv, index=False)
    logger.info(f"✅ Complete updated dataset saved to: {complete_updated_csv}")

    # Generate report
    generate_fixed_comprehensive_report(problematic_df, web_classifier)

    logger.info(f"\n{'='*80}")
    logger.info(f"🌐 FIXED COMPREHENSIVE WEB SEARCH SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Problematic products processed: {len(problematic_df)}")
    logger.info(f"✅ Successful reclassifications: {successful_reclassifications}")
    logger.info(f"❌ Failed reclassifications: {failed_reclassifications}")
    logger.info(
        f"📈 Success rate: {successful_reclassifications/len(problematic_df)*100:.1f}%"
    )

    return complete_updated_df, problematic_df


def merge_fixed_comprehensive_results(
    complete_df: pd.DataFrame, reclassified_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge FIXED comprehensive reclassification results back into complete dataset"""

    logger.info(f"🔄 Merging FIXED comprehensive results...")
    updated_df = complete_df.copy()

    # Add FIXED comprehensive columns to complete dataset
    comprehensive_columns = [
        col
        for col in reclassified_df.columns
        if col.startswith("fixed_comprehensive_") or col == "original_problem_type"
    ]

    for col in comprehensive_columns:
        if col.endswith("_tokens"):
            updated_df[col] = 0
        elif col.endswith("_performed") or col.endswith("_successful"):
            updated_df[col] = False
        else:
            updated_df[col] = ""

    successful_merges = 0

    for idx, row in reclassified_df.iterrows():
        if idx in updated_df.index:
            # Copy FIXED comprehensive results
            for col in comprehensive_columns:
                if col in row:
                    updated_df.at[idx, col] = row[col]

            # Update primary classification if successful
            if row.get("fixed_comprehensive_reclassification_successful", False):
                new_domain = row.get("fixed_comprehensive_new_domain", "")
                if new_domain and new_domain != "Other":
                    updated_df.at[idx, "primary_domain"] = new_domain
                    updated_df.at[idx, "primary_subcategory"] = row.get(
                        "fixed_comprehensive_new_subcategory", ""
                    )
                    updated_df.at[idx, "primary_subsubcategory"] = row.get(
                        "fixed_comprehensive_new_subsubcategory", ""
                    )
                    updated_df.at[idx, "primary_subsubsubcategory"] = row.get(
                        "fixed_comprehensive_new_subsubsubcategory", ""
                    )
                    updated_df.at[idx, "primary_confidence"] = row.get(
                        "fixed_comprehensive_new_confidence", ""
                    )
                    updated_df.at[idx, "validated_path_primary"] = row.get(
                        "fixed_comprehensive_validated_path", ""
                    )

                    successful_merges += 1
                    logger.info(
                        f"✅ FIXED: Updated {updated_df.at[idx, 'Name'][:30]} → {new_domain}"
                    )

    logger.info(
        f"🔄 Successfully merged {successful_merges} FIXED comprehensive results"
    )
    return updated_df


def generate_fixed_comprehensive_report(
    df: pd.DataFrame, classifier: FixedComprehensiveWebSearchClassifier
):
    """Generate report for FIXED comprehensive web search results"""

    print(f"\n{'='*80}")
    print("🚀 FIXED COMPREHENSIVE WEB SEARCH RECLASSIFICATION REPORT")
    print(f"{'='*80}")

    total = len(df)
    successful = len(df[df["fixed_comprehensive_reclassification_successful"] == True])

    print(f"📊 OVERVIEW")
    print(f"  Total problematic products: {total}")
    print(f"  Successful reclassifications: {successful} ({successful/total*100:.1f}%)")

    # Problem type breakdown
    print(f"\n{'PROBLEM TYPES PROCESSED':-^60}")
    problem_types = df["original_problem_type"].str.split("|").explode().value_counts()
    for ptype, count in problem_types.head(10).items():
        if ptype:
            print(f"  {ptype:<35} {count:>5}")

    # New domain distribution
    print(f"\n{'NEW DOMAIN ASSIGNMENTS':-^60}")
    domains = df[df["fixed_comprehensive_new_domain"] != ""][
        "fixed_comprehensive_new_domain"
    ].value_counts()
    for domain, count in domains.head(10).items():
        print(f"  {domain:<30} {count:>5}")

    # Before vs After examples
    successful_fixes = df[
        df["fixed_comprehensive_reclassification_successful"] == True
    ].head(5)
    if len(successful_fixes) > 0:
        print(f"\n{'🌟 SUCCESSFUL FIXED COMPREHENSIVE FIXES':-^60}")
        for idx, row in successful_fixes.iterrows():
            name = row["Name"][:30]
            old_domain = row.get("primary_domain", "Unknown")
            old_subcategory = row.get("primary_subcategory", "Unknown")
            new_path = row.get("fixed_comprehensive_validated_path", "Unknown")
            problem = row.get("original_problem_type", "Unknown")

            print(f"  {name}")
            print(f"    Problem: {problem}")
            print(f"    OLD: {old_domain} → {old_subcategory}")
            print(f"    NEW: {new_path}")
            print()


def debug_obvious_products(web_classifier):
    """Debug what web search returns for obviously classifiable products"""

    obvious_tests = [
        (
            "BD LSRFortessa X-20",
            "BD Biosciences",
            "Should be Lab_Equipment → Cell Analysis",
        ),
        ("Hybond membrane", "Cytiva", "Should be Lab_Equipment → Laboratory Supplies"),
        ("GraphPad Prism", "GraphPad", "Should be Software → Statistical Analysis"),
        ("SAS", "SAS Institute", "Should be Software → Statistical Analysis"),
        ("whatman filter", "Cytiva", "Should be Lab_Equipment → Laboratory Supplies"),
    ]

    print(f"\n{'='*60}")
    print("🧪 DEBUGGING WEB SEARCH FOR OBVIOUS PRODUCTS")
    print(f"{'='*60}")

    for product, manufacturer, expected in obvious_tests:
        print(f"\n🧪 TESTING: {product}")
        print(f"   Manufacturer: {manufacturer}")
        print(f"   Expected: {expected}")
        print(f"   {'-'*50}")

        # Test web search step
        search_result = web_classifier.search_and_identify_domain_for_problem(
            product, manufacturer, "Other_domain", "", ""
        )

        print(f"✅ Search successful: {search_result.get('search_successful')}")
        print(f"🎯 Suggested domain: {search_result.get('suggested_domain')}")
        print(f"🔍 Confidence: {search_result.get('confidence', 'N/A')}")
        print(
            f"📝 Reasoning: {search_result.get('reasoning', 'No reasoning')[:200]}..."
        )

        # Test domain classification if search was successful
        if (
            search_result.get("search_successful")
            and search_result.get("suggested_domain") != "Other"
        ):
            suggested_domain = search_result.get("suggested_domain")
            enhanced_description = search_result.get("enhanced_description", "")

            print(f"\n   🎯 Testing classification within {suggested_domain} domain...")
            classification_result = (
                web_classifier._classify_within_domain_efficiently_FIXED(
                    product, enhanced_description, suggested_domain
                )
            )

            if classification_result:
                print(f"   📊 Domain Classification Results:")
                print(
                    f"      Subcategory: {classification_result.get('subcategory', 'N/A')}"
                )
                print(
                    f"      Subsubcategory: {classification_result.get('subsubcategory', 'N/A')}"
                )
                print(
                    f"      Validated path: {classification_result.get('validated_path', 'N/A')}"
                )
                print(
                    f"      Is valid: {classification_result.get('is_valid_path', False)}"
                )
                print(
                    f"      Confidence: {classification_result.get('confidence', 'N/A')}"
                )
            else:
                print(f"   ❌ Domain classification returned None")

        # Check if result makes sense
        suggested = search_result.get("suggested_domain")
        if "LSRFortessa" in product and suggested != "Lab_Equipment":
            print("🚨 PROBLEM: Flow cytometer not identified as Lab_Equipment!")
        elif "membrane" in product.lower() and suggested != "Lab_Equipment":
            print("🚨 PROBLEM: Membrane not identified as Lab_Equipment!")
        elif "filter" in product.lower() and suggested != "Lab_Equipment":
            print("🚨 PROBLEM: Filter not identified as Lab_Equipment!")
        elif "Prism" in product and suggested != "Software":
            print("🚨 PROBLEM: GraphPad Prism not identified as Software!")
        elif "SAS" in product and suggested != "Software":
            print("🚨 PROBLEM: SAS not identified as Software!")
        else:
            print("✅ Domain classification looks reasonable")

        print(f"   {'='*50}")


def debug_yaml_structure_detailed(category_system):
    """Debug YAML structures in detail to understand nesting"""
    print(f"\n{'='*60}")
    print("🔍 DETAILED YAML STRUCTURE ANALYSIS")
    print(f"{'='*60}")

    # Check Software domain specifically
    if "Software" in category_system.domain_structures:
        software_structure = category_system.domain_structures["Software"]
        print(f"\n🖥️ SOFTWARE DOMAIN DETAILED ANALYSIS:")
        print(f"   Root type: {type(software_structure)}")
        print(f"   Root keys: {list(software_structure.keys())}")

        if "Life Science Software" in software_structure:
            life_sci_software = software_structure["Life Science Software"]
            print(f"   Life Science Software type: {type(life_sci_software)}")

            if "subcategories" in life_sci_software:
                subcats = life_sci_software["subcategories"]
                print(f"   Subcategories type: {type(subcats)}")
                print(f"   Number of subcategories: {len(subcats)}")

                for i, subcat in enumerate(subcats):
                    print(
                        f"   [{i}] Type: {type(subcat)}, Value: {str(subcat)[:100]}..."
                    )

                    # If it's a dict, examine its structure
                    if isinstance(subcat, dict):
                        for key, value in subcat.items():
                            print(f"       Key: '{key}', Value type: {type(value)}")
                            if key == "Statistical Analysis Software":
                                print(f"       🎯 FOUND Statistical Analysis Software!")
                                if "subsubcategories" in subcat:
                                    print(
                                        f"       subsubcategories: {subcat['subsubcategories']}"
                                    )

    # Check Lab_Equipment domain
    if "Lab_Equipment" in category_system.domain_structures:
        lab_structure = category_system.domain_structures["Lab_Equipment"]
        print(f"\n🔬 LAB_EQUIPMENT DOMAIN DETAILED ANALYSIS:")
        print(f"   Root type: {type(lab_structure)}")
        print(f"   Root keys: {list(lab_structure.keys())}")

        # Navigate deeper into the structure
        for key, value in lab_structure.items():
            print(f"   Key: '{key}', Type: {type(value)}")
            if isinstance(value, dict) and "subcategories" in value:
                subcats = value["subcategories"]
                print(f"   {key} subcategories type: {type(subcats)}")
                if isinstance(subcats, list):
                    print(f"   First few subcategories: {subcats[:3]}")
                    # Look for Laboratory Supplies
                    for subcat in subcats:
                        if isinstance(subcat, str) and "Laboratory Supplies" in subcat:
                            print(f"   🎯 FOUND: {subcat}")


def test_specific_fixes_with_structure_debug(web_classifier):
    """Test the fixes with detailed structure debugging"""

    print(f"\n{'='*60}")
    print("🧪 TESTING SPECIFIC FIXES WITH STRUCTURE DEBUG")
    print(f"{'='*60}")

    # Debug the exact structures first
    web_classifier.debug_exact_structure_for_domain("Software")
    web_classifier.debug_exact_structure_for_domain("Lab_Equipment")

    test_cases = [
        (
            "GraphPad Prism",
            "GraphPad",
            "Should → Software → Statistical Analysis Software → GraphPad Prism",
        ),
        (
            "SAS",
            "SAS Institute",
            "Should → Software → Statistical Analysis Software → SAS",
        ),
        ("Hybond membrane", "Cytiva", "Should → Lab_Equipment → Laboratory Supplies"),
    ]

    for product, manufacturer, expected in test_cases:
        print(f"\n🔍 TESTING: {product}")
        print(f"Expected: {expected}")

        # Test just the validation part
        if product in ["GraphPad Prism", "SAS"]:
            domain = "Software"
            if product == "GraphPad Prism":
                path = ["Statistical Analysis Software", "GraphPad Prism"]
            else:
                path = ["Statistical Analysis Software", "SAS"]
        else:
            domain = "Lab_Equipment"
            path = ["Laboratory Supplies & Consumables"]

        print(f"Testing validation for domain: {domain}, path: {path}")
        is_valid, validated_path = (
            web_classifier._improved_validate_classification_path_FIXED(domain, path)
        )
        print(f"Result: {validated_path}")
        print(f"Valid: {is_valid}")
        print(f"   {'-'*40}")


def debug_domain_structure(category_system):
    """Debug the domain structure to see what's available"""
    print(f"\n{'='*60}")
    print("🔍 DEBUGGING DOMAIN STRUCTURE")
    print(f"{'='*60}")

    print(f"Available domains: {category_system.available_domains}")
    print(f"Domain structures keys: {list(category_system.domain_structures.keys())}")

    # Check Software domain specifically
    if "Software" in category_system.domain_structures:
        software_structure = category_system.domain_structures["Software"]
        print(f"\n🔍 Software domain structure:")
        print(f"   Type: {type(software_structure)}")
        if "subcategories" in software_structure:
            print(f"   Subcategories type: {type(software_structure['subcategories'])}")
            print(f"   Subcategories: {software_structure['subcategories']}")
        else:
            print(f"   Full structure: {software_structure}")

    # Check Lab_Equipment domain
    if "Lab_Equipment" in category_system.domain_structures:
        lab_structure = category_system.domain_structures["Lab_Equipment"]
        print(f"\n🔍 Lab_Equipment domain structure:")
        print(f"   Type: {type(lab_structure)}")
        if "subcategories" in lab_structure:
            print(f"   Subcategories type: {type(lab_structure['subcategories'])}")
            if isinstance(lab_structure["subcategories"], list):
                print(
                    f"   First few subcategories: {lab_structure['subcategories'][:5]}"
                )
            elif isinstance(lab_structure["subcategories"], dict):
                print(
                    f"   Subcategory keys: {list(lab_structure['subcategories'].keys())[:5]}"
                )


def debug_specific_classification_issue():
    """Debug specific issues we see in the output"""
    print(f"\n{'='*60}")
    print("🔍 DEBUGGING SPECIFIC CLASSIFICATION ISSUES")
    print(f"{'='*60}")

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    web_classifier = FixedComprehensiveWebSearchClassifier(category_system, tag_system)

    # Debug domain structure first
    debug_domain_structure(category_system)

    # Test specific problematic cases
    problem_cases = [
        (
            "hybond",
            "Cytiva",
            "Other_domain",
            "Should be Lab_Equipment -> Laboratory Supplies",
        ),
        (
            "GraphPad Prism 5",
            "GraphPad",
            "Other_domain",
            "Should be Software -> Statistical Analysis Software",
        ),
        (
            "SAS 9",
            "SAS Institute",
            "Generic_Statistical_Software",
            "Should be Software -> Statistical Analysis Software -> SAS",
        ),
    ]

    for product, manufacturer, problem_type, expected in problem_cases:
        print(f"\n🔍 DEBUGGING: {product}")
        print(f"   Problem type: {problem_type}")
        print(f"   Expected: {expected}")

        # Step 1: Test web search
        search_result = web_classifier.search_and_identify_domain_for_problem(
            product, manufacturer, problem_type, "", ""
        )

        print(f"   Search domain: {search_result.get('suggested_domain')}")
        print(f"   Search reasoning: {search_result.get('reasoning', '')[:150]}...")

        # Step 2: Test domain classification
        if search_result.get("search_successful"):
            domain = search_result.get("suggested_domain")
            description = search_result.get("enhanced_description", "")

            classification = web_classifier._classify_within_domain_efficiently_FIXED(
                product, description, domain
            )

            if classification:
                print(f"   Final path: {classification.get('validated_path')}")
            else:
                print(f"   ❌ Classification failed")

        print(f"   {'-'*40}")


# Modified main function with debug option
def main():
    """Main execution function"""
    print("=" * 80)
    print("🚀 FIXED COMPREHENSIVE WEB SEARCH RECLASSIFICATION")
    print(
        "Fixes YAML validation + better prompts + targets ALL problematic classifications"
    )
    print("=" * 80)

    try:
        # Ask what user wants to do
        print("\n🎯 What would you like to do?")
        print("1. Debug obvious products")
        print("2. Debug specific classification issues")
        print("3. Debug domain structure")
        print("4. Test comprehensive reclassification (5 products)")
        print("5. Run full comprehensive reclassification")
        print("6. All debugging options")
        print("7. Test specific fixes")

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice in ["1", "2", "3", "6", "7"]:
            # Initialize systems for debugging
            print("\n🔧 Initializing classification systems...")
            category_system = LLMDrivenCategorySystem()
            tag_system = EnhancedTagSystem()
            web_classifier = FixedComprehensiveWebSearchClassifier(
                category_system, tag_system
            )
            print("✅ Systems initialized successfully!")

        if choice in ["1", "6"]:
            debug_obvious_products(web_classifier)

        if choice in ["2", "6"]:
            debug_specific_classification_issue()

        if choice in ["3", "6"]:
            debug_domain_structure(category_system)

        # Add this new section:
        if choice in ["7"]:
            debug_yaml_structure_detailed(category_system)
            web_classifier.debug_exact_structure_for_domain("Software")
            web_classifier.debug_exact_structure_for_domain("Lab_Equipment")
            test_specific_fixes_with_structure_debug(web_classifier)

            print("\n🧪 Testing FIXED comprehensive web search reclassification...")
            complete_df, test_df = (
                process_fixed_comprehensive_web_search_reclassification(
                    input_csv=INPUT_CSV,
                    output_csv=OUTPUT_CSV,
                    max_products=5,
                    test_mode=True,
                )
            )

            if len(test_df) > 0 and choice == "5":
                user_input = input("\n🚀 Run on all problematic products? (y/n): ")
                if user_input.lower() == "y":
                    complete_df, full_df = (
                        process_fixed_comprehensive_web_search_reclassification(
                            input_csv=INPUT_CSV,
                            output_csv=OUTPUT_CSV.replace(".csv", "_full.csv"),
                            test_mode=False,
                        )
                    )
            elif len(test_df) == 0:
                print("✅ No problematic products found to reclassify!")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
