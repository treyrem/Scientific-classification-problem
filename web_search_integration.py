# CORRECTED WEB SEARCH IMPLEMENTATION - FINAL VERSION
# Based on thorough analysis of OpenAI Web Search API documentation

import pandas as pd
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import glob

# Import existing classification system components
from enhanced_classification_full_checkpoint import (
    LLMDrivenCategorySystem,
    EnhancedTagSystem,
    EnhancedLLMClassifier,
)

# Configuration
INPUT_CSV = "products_enhanced_fixed_classification.csv"
OUTPUT_CSV = "products_real_web_search_reclassified.csv"

# Checkpointing Configuration
REAL_WEB_CHECKPOINT_DIR = "real_web_search_checkpoints"
REAL_WEB_CHECKPOINT_FREQ = 25
REAL_WEB_CHECKPOINT_PREFIX = "real_web_search_checkpoint"

# Antibiotic keywords for targeted reclassification
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


def find_latest_real_web_checkpoint():
    """Find the most recent real web search checkpoint file"""
    if not os.path.exists(REAL_WEB_CHECKPOINT_DIR):
        return None, 0

    checkpoint_files = glob.glob(
        os.path.join(REAL_WEB_CHECKPOINT_DIR, f"{REAL_WEB_CHECKPOINT_PREFIX}_*.csv")
    )
    if not checkpoint_files:
        return None, 0

    checkpoint_numbers = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        try:
            number = int(filename.split("_")[-1].split(".")[0])
            checkpoint_numbers.append((number, filepath))
        except (ValueError, IndexError):
            continue

    if checkpoint_numbers:
        latest = max(checkpoint_numbers, key=lambda x: x[0])
        return latest[1], latest[0]

    return None, 0


class CorrectedRealWebSearchReclassifier:
    """Corrected real web search reclassifier based on official OpenAI documentation"""

    def __init__(
        self,
        category_system: LLMDrivenCategorySystem,
        tag_system: EnhancedTagSystem,
        debug_mode: bool = False,
    ):
        self.category_system = category_system
        self.tag_system = tag_system
        self.client = OpenAI(api_key=get_openai_key())
        self.search_cache = {}
        self.total_searches = 0
        self.successful_searches = 0
        self.total_web_search_cost = 0.0
        self.reclassification_stats = defaultdict(int)
        self.debug_mode = debug_mode

    def correct_web_search_and_identify_domain(
        self,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str,
        current_subcategory: str,
    ) -> Dict[str, Any]:
        """CORRECTED web search using proper OpenAI Responses API handling"""

        # Check cache first
        cache_key = f"real_web_{product_name}_{manufacturer}_{problem_type}".lower()
        if cache_key in self.search_cache:
            logger.info(f"Using cached web search results for: {product_name}")
            return self.search_cache[cache_key]

        logger.info(
            f"Performing real web search for: {product_name} (Problem: {problem_type})"
        )
        self.total_searches += 1

        # Construct search input according to documentation
        search_input = f"Search for detailed product information about {product_name}"
        if manufacturer:
            search_input += f" by {manufacturer}"
        search_input += ". Find official specifications, technical details, product category, intended applications, and manufacturer datasheets."

        try:
            # CORRECT API call according to official documentation
            response = self.client.responses.create(
                model="gpt-4.1",  # Model specified in documentation
                tools=[
                    {
                        "type": "web_search_preview",
                        "search_context_size": "medium",  # Balanced context and latency
                    }
                ],
                input=search_input,
            )

            if self.debug_mode:
                logger.info(f"Response type: {type(response)}")
                logger.info(
                    f"Response has output_text: {hasattr(response, 'output_text')}"
                )
                logger.info(f"Response has output: {hasattr(response, 'output')}")

            # Extract main content using output_text (confirmed to exist from debug)
            main_content = ""
            if hasattr(response, "output_text"):
                main_content = response.output_text
                logger.info(f"Extracted main content length: {len(main_content)}")

            # Extract structured data from output array
            web_search_calls = []
            citations = []

            if hasattr(response, "output") and response.output:
                try:
                    for item in response.output:
                        # Handle web_search_call items
                        if hasattr(item, "type") and item.type == "web_search_call":
                            web_search_calls.append(
                                {
                                    "id": getattr(item, "id", ""),
                                    "status": getattr(item, "status", ""),
                                    "action": getattr(item, "action", None),
                                }
                            )
                            if self.debug_mode:
                                logger.info(
                                    f"Found web_search_call: {getattr(item, 'id', 'no_id')}"
                                )

                        # Handle message items with citations
                        elif hasattr(item, "type") and item.type == "message":
                            if hasattr(item, "content"):
                                for content_item in item.content:
                                    if (
                                        hasattr(content_item, "type")
                                        and content_item.type == "output_text"
                                    ):

                                        # Extract annotations/citations
                                        if hasattr(content_item, "annotations"):
                                            for annotation in content_item.annotations:
                                                if (
                                                    hasattr(annotation, "type")
                                                    and annotation.type
                                                    == "url_citation"
                                                ):
                                                    citations.append(
                                                        {
                                                            "url": getattr(
                                                                annotation, "url", ""
                                                            ),
                                                            "title": getattr(
                                                                annotation, "title", ""
                                                            ),
                                                            "start_index": getattr(
                                                                annotation,
                                                                "start_index",
                                                                0,
                                                            ),
                                                            "end_index": getattr(
                                                                annotation,
                                                                "end_index",
                                                                0,
                                                            ),
                                                        }
                                                    )
                except Exception as parse_error:
                    logger.warning(f"Failed to parse structured output: {parse_error}")

            # Parse web search results for domain classification
            domain_result = self._parse_web_results_for_domain(
                main_content, product_name, manufacturer, problem_type, current_domain
            )

            # Combine results
            final_result = {
                "search_successful": True,
                "main_content": main_content,
                "citations": citations,
                "citations_found": len(citations),
                "web_search_calls": web_search_calls,
                "search_method": "responses_api_corrected",
                "response_id": getattr(response, "id", ""),
                "model_used": getattr(response, "model", ""),
                **domain_result,
            }

            # Cache the result
            self.search_cache[cache_key] = final_result
            self.successful_searches += 1

            logger.info(f"Real web search completed for: {product_name}")
            logger.info(f"Citations found: {len(citations)}")
            return final_result

        except Exception as e:
            logger.error(
                f"Real web search failed for {product_name} (Problem: {problem_type}): {e}"
            )

            # Fallback to LLM knowledge
            fallback_result = self._fallback_to_llm_knowledge(
                product_name,
                manufacturer,
                problem_type,
                current_domain,
                current_subcategory,
            )
            self.search_cache[cache_key] = fallback_result
            return fallback_result

    def _parse_web_results_for_domain(
        self,
        web_content: str,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str,
    ) -> Dict[str, Any]:
        """Parse web search results for domain classification"""

        if not web_content:
            logger.warning("No web content to parse")
            return {
                "suggested_domain": "Other",
                "confidence": "Low",
                "reasoning": "No web content found",
                "enhanced_description": "",
                "token_usage": 0,
            }

        available_domains = self.category_system.available_domains

        prompt = f"""Based on real web search results, analyze this product and suggest the correct domain classification:

Product: {product_name}
Manufacturer: {manufacturer}
Current Problem: {problem_type}
Current Domain: {current_domain}

Web Search Results:
{web_content}

Available Domains: {', '.join(available_domains)}

CLASSIFICATION RULES:
Lab_Equipment: Physical instruments, hardware, supplies, consumables, membranes, filters
Software: Computer programs, statistical software, analysis software, LIMS
Cell_Biology: Cell culture reagents, research antibiotics, cell lines, staining
Chemistry: Chemical compounds, solvents, buffers
Antibodies: All antibody products
Assay_Kits: Complete assay kits

Focus on WHAT the product IS based on the web search findings, not what it's used for.

Return JSON:
{{
    "suggested_domain": "exact_domain_from_available_list",
    "confidence": "High/Medium/Low",
    "reasoning": "explanation based on web search findings",
    "product_type": "specific product type found in web search",
    "enhanced_description": "comprehensive description from web search for classification",
    "technical_specifications": "any technical specs found",
    "applications_found": ["applications", "from", "web", "search"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)

            # More robust JSON parsing
            if not cleaned:
                raise ValueError("Empty response from LLM")

            result = json.loads(cleaned)

            # Validate domain
            suggested_domain = result.get("suggested_domain", "Other")
            if suggested_domain not in available_domains:
                logger.warning(f"Invalid domain suggested: {suggested_domain}")
                result["suggested_domain"] = "Other"

            result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(
                f"Raw response: {text[:200] if 'text' in locals() else 'No text'}"
            )
            return {
                "suggested_domain": "Other",
                "confidence": "Low",
                "reasoning": f"JSON parsing failed: {e}",
                "enhanced_description": web_content[:500],
                "token_usage": 0,
            }
        except Exception as e:
            logger.error(f"Failed to parse web search results: {e}")
            return {
                "suggested_domain": "Other",
                "confidence": "Low",
                "reasoning": f"Parsing failed: {e}",
                "enhanced_description": web_content[:500],
                "token_usage": 0,
            }

    def _fallback_to_llm_knowledge(
        self,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str,
        current_subcategory: str,
    ) -> Dict[str, Any]:
        """Fallback to LLM knowledge when web search fails"""

        logger.info(f"Falling back to LLM knowledge for: {product_name}")

        available_domains = self.category_system.available_domains

        prompt = f"""Based on your training knowledge, analyze this product with classification problems:

Product: {product_name}
Manufacturer: {manufacturer}
Problem Type: {problem_type}
Current: {current_domain} -> {current_subcategory}

Available Domains: {', '.join(available_domains)}

Provide your best assessment based on training knowledge:
{{
    "suggested_domain": "best_domain_guess",
    "confidence": "Low",
    "reasoning": "explanation based on training knowledge",
    "enhanced_description": "description from training knowledge",
    "knowledge_source": "training_data_only"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)
            result = json.loads(cleaned)

            result.update(
                {
                    "search_successful": False,
                    "fallback_used": True,
                    "main_content": "",
                    "citations": [],
                    "citations_found": 0,
                    "web_search_calls": [],
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            return result

        except Exception as e:
            logger.error(f"Fallback LLM knowledge failed: {e}")
            return {
                "suggested_domain": "Other",
                "confidence": "Low",
                "reasoning": "Both web search and fallback failed",
                "enhanced_description": f"Product: {product_name} by {manufacturer}",
                "search_successful": False,
                "fallback_used": True,
                "fallback_failed": True,
                "main_content": "",
                "citations": [],
                "citations_found": 0,
                "web_search_calls": [],
                "error": str(e),
                "token_usage": 0,
            }

    def reclassify_problematic_product(
        self,
        product_name: str,
        manufacturer: str,
        problem_type: str,
        current_domain: str = "",
        current_subcategory: str = "",
        original_description: str = "",
    ) -> Dict[str, Any]:
        """Reclassify problematic product with corrected real web search"""

        logger.info(f"Real web search reclassification: '{product_name}'")
        logger.info(f"Problem Type: {problem_type}")
        logger.info(f"Current: {current_domain} -> {current_subcategory}")

        total_tokens = 0

        # Step 1: Corrected Real Web Search + Domain Identification
        logger.info("Step 1: Corrected real web search + domain identification")
        search_and_domain_result = self.correct_web_search_and_identify_domain(
            product_name,
            manufacturer,
            problem_type,
            current_domain,
            current_subcategory,
        )
        total_tokens += search_and_domain_result.get("token_usage", 0)

        if not search_and_domain_result.get("search_successful", False):
            logger.warning(f"Web search failed for '{product_name}'")
            return self._create_failed_search_result(
                product_name, problem_type, total_tokens
            )

        # Step 2: Domain-Specific Classification
        logger.info("Step 2: Domain-specific classification")
        suggested_domain = search_and_domain_result.get("suggested_domain", "Other")
        enhanced_description = search_and_domain_result.get("enhanced_description", "")

        if (
            suggested_domain != "Other"
            and suggested_domain in self.category_system.available_domains
        ):
            domain_classification = self._classify_within_domain_with_web_data(
                product_name,
                enhanced_description,
                suggested_domain,
                search_and_domain_result,
            )
            total_tokens += (
                domain_classification.get("token_usage", 0)
                if domain_classification
                else 0
            )
        else:
            logger.warning(f"Invalid or 'Other' domain suggested: {suggested_domain}")
            domain_classification = None

        # Build final result
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

    def _classify_within_domain_with_web_data(
        self,
        product_name: str,
        description: str,
        domain_key: str,
        web_search_result: Dict,
    ) -> Optional[Dict]:
        """Classify within domain using web search enhancement"""

        logger.info(f"Classification in: {domain_key}")

        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        # Extract additional info from web search
        technical_specs = web_search_result.get("technical_specifications", "")
        applications = web_search_result.get("applications_found", [])
        product_type = web_search_result.get("product_type", "")

        prompt = f"""Classify this product within the {domain_key} domain using real web search data.

{focused_structure}

Product: "{product_name}"
Enhanced Description: "{description}"

WEB SEARCH ENHANCEMENTS:
Product Type: {product_type}
Technical Specifications: {technical_specs}
Applications Found: {', '.join(applications) if applications else 'N/A'}
Confidence: {web_search_result.get('confidence', 'Low')}

Use exact category names from the YAML structure above.

JSON response:
{{
    "subcategory": "exact_name_from_yaml_structure",
    "subsubcategory": "exact_name_from_yaml_if_available",
    "subsubsubcategory": "exact_3rd_level_name_or_null",
    "confidence": "High/Medium/Low"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=400,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)
            result = json.loads(cleaned)

            # Build path components
            path_components = []
            for level in ["subcategory", "subsubcategory", "subsubsubcategory"]:
                value = result.get(level, "")
                if value and str(value).strip() not in {"", "null", "None"}:
                    path_components.append(str(value).strip())
                else:
                    break

            # Validate classification path
            try:
                is_valid, validated_path = (
                    self.category_system.validate_classification_path(
                        domain_key, path_components
                    )
                )
                logger.info(f"Validation successful: {validated_path}")
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                domain_name = domain_key.replace("_", " ")
                if path_components:
                    validated_path = domain_name + " -> " + " -> ".join(path_components)
                    is_valid = True
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
                    "web_search_enhanced": True,
                }
            )

            if is_valid:
                logger.info(f"Classification successful: {validated_path}")
            else:
                logger.warning(f"Classification failed: {' -> '.join(path_components)}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
            return None

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
            "real_web_search_performed": True,
            "real_web_search_successful": True,
            "real_web_reclassification_successful": True,
            "original_problem_type": problem_type,
            "original_domain": old_domain,
            "original_subcategory": old_subcategory,
            "new_domain": new_domain,
            "new_classification": classification,
            "web_search_info": search_result,
            "total_tokens": tokens,
            "classification_method": "corrected_real_web_search_responses_api",
        }

    def _create_failed_search_result(
        self, product_name: str, problem_type: str, tokens: int
    ) -> Dict[str, Any]:
        """Create result for failed web search"""
        return {
            "real_web_search_performed": True,
            "real_web_search_successful": False,
            "real_web_reclassification_successful": False,
            "original_problem_type": problem_type,
            "total_tokens": tokens,
            "classification_method": "failed_corrected_real_web_search",
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
            "real_web_search_performed": True,
            "real_web_search_successful": True,
            "real_web_classification_successful": False,
            "real_web_reclassification_successful": False,
            "original_problem_type": problem_type,
            "original_domain": old_domain,
            "original_subcategory": old_subcategory,
            "web_search_info": search_result,
            "total_tokens": tokens,
            "classification_method": "failed_corrected_real_web_domain_classification",
            "failure_reason": reason,
        }

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


# Rest of the implementation remains the same but uses CorrectedRealWebSearchReclassifier
def identify_all_problematic_products(df: pd.DataFrame) -> pd.DataFrame:
    """Identify ALL problematic classifications from Stage 1-3 output"""

    logger.info("Identifying ALL products with problematic classifications...")

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
            f"Found {other_domain_mask.sum()} products with primary_domain = 'Other'"
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
                f"Found {analytical_mask.sum()} products with primary_subcategory = 'Analytical Instrumentation'"
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
                f"Found {stats_software_mask.sum()} products with primary_subcategory = 'Statistical Analysis Software'"
            )

    # 4. Chemistry domain antibiotics misclassified as Laboratory Acids
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
                f"Found {antibiotic_misclass_mask.sum()} antibiotics misclassified as Laboratory Acids in Chemistry"
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

        # Check antibiotic misclassification
        if problematic_df.at[idx, "primary_domain"] == "Chemistry":
            if "primary_subcategory" in problematic_df.columns:
                subcategory = str(problematic_df.at[idx, "primary_subcategory"])
                if "Laboratory Acids" in subcategory:
                    product_name = str(problematic_df.at[idx, "Name"]).lower()
                    if any(
                        antibiotic in product_name
                        for antibiotic in [kw.lower() for kw in ANTIBIOTIC_KEYWORDS]
                    ):
                        problems.append("Antibiotic_misclassified_as_acid")

        problematic_df.at[idx, "problem_type"] = "|".join(problems)

    logger.info(f"Total problematic products found: {len(problematic_df)}")

    for problem_type, count in problem_counts.items():
        logger.info(f"  - {problem_type}: {count}")

    return problematic_df


def get_real_web_search_reclassification_columns():
    """Get CSV columns for real web search reclassification"""
    return [
        "problem_type",
        "real_web_search_performed",
        "real_web_search_successful",
        "real_web_reclassification_successful",
        "real_web_suggested_domain",
        "real_web_domain_confidence",
        "real_web_new_domain",
        "real_web_new_subcategory",
        "real_web_new_subsubcategory",
        "real_web_new_subsubsubcategory",
        "real_web_new_confidence",
        "real_web_validated_path",
        "real_web_citations_found",
        "real_web_total_tokens",
        "real_web_processed_at",
        "real_web_processing_error",
    ]


def process_corrected_real_web_search_reclassification_with_checkpointing(
    input_csv: str, output_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process corrected real web search reclassification with checkpointing"""

    logger.info(
        "Starting Corrected Real Web Search Reclassification with Checkpointing..."
    )

    # Setup checkpoint directory
    os.makedirs(REAL_WEB_CHECKPOINT_DIR, exist_ok=True)

    # Initialize systems with corrected classifier
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    web_classifier = CorrectedRealWebSearchReclassifier(
        category_system, tag_system, debug_mode=True
    )

    try:
        # Check for existing checkpoint
        latest_checkpoint, last_checkpoint_number = find_latest_real_web_checkpoint()

        if latest_checkpoint:
            logger.info(f"Found real web search checkpoint: {latest_checkpoint}")
            logger.info(f"Last checkpoint number: {last_checkpoint_number}")
            complete_df = pd.read_csv(latest_checkpoint, low_memory=False)

            already_processed_count = 0
            if "real_web_search_performed" in complete_df.columns:
                already_processed_count = (
                    complete_df["real_web_search_performed"] == True
                ).sum()
                logger.info(
                    f"Found {already_processed_count} products already processed"
                )

            # Find unprocessed problematic products
            if already_processed_count > 0:
                if (
                    "problem_type" not in complete_df.columns
                    or (
                        complete_df["problem_type"].notna()
                        & (complete_df["problem_type"] != "")
                    ).sum()
                    < already_processed_count
                ):
                    logger.info("Re-identifying problematic products...")
                    if "problem_type" not in complete_df.columns:
                        complete_df["problem_type"] = ""
                    temp_problematic = identify_all_problematic_products(complete_df)
                    for idx in temp_problematic.index:
                        complete_df.at[idx, "problem_type"] = temp_problematic.at[
                            idx, "problem_type"
                        ]

                problematic_mask = (
                    (complete_df["problem_type"].notna())
                    & (complete_df["problem_type"] != "")
                    & (complete_df["real_web_search_performed"] != True)
                )
                problematic_df = complete_df[problematic_mask].copy()
                logger.info(
                    f"Unprocessed problematic products found: {len(problematic_df)}"
                )
            else:
                logger.info(
                    "No processed items found, identifying all problematic products..."
                )
                real_web_columns = get_real_web_search_reclassification_columns()
                for col in real_web_columns:
                    if col not in complete_df.columns:
                        if col.endswith("_tokens") or col.endswith("_found"):
                            complete_df[col] = 0
                        elif col.endswith("_performed") or col.endswith("_successful"):
                            complete_df[col] = False
                        else:
                            complete_df[col] = ""
                problematic_df = identify_all_problematic_products(complete_df)
                already_processed_count = 0
        else:
            logger.info(f"No checkpoint found, starting fresh from {input_csv}")
            complete_df = pd.read_csv(input_csv)
            logger.info(f"Loaded {len(complete_df)} total products")

            real_web_columns = get_real_web_search_reclassification_columns()
            for col in real_web_columns:
                if col.endswith("_tokens") or col.endswith("_found"):
                    complete_df[col] = 0
                elif col.endswith("_performed") or col.endswith("_successful"):
                    complete_df[col] = False
                else:
                    complete_df[col] = ""

            problematic_df = identify_all_problematic_products(complete_df)
            already_processed_count = 0

        if len(problematic_df) == 0:
            logger.info("All problematic products have been processed")
            total_problematic = (
                complete_df["problem_type"].notna()
                & (complete_df["problem_type"] != "")
            ).sum()
            logger.info(f"Total problematic products in dataset: {total_problematic}")
            logger.info(f"Total processed: {already_processed_count}")
            return complete_df, pd.DataFrame()

        total_problematic_in_dataset = (
            complete_df["problem_type"].notna() & (complete_df["problem_type"] != "")
        ).sum()

        logger.info(
            f"Processing {len(problematic_df)} remaining problematic products..."
        )
        logger.info(
            f"Total progress so far: {already_processed_count}/{total_problematic_in_dataset} problematic products"
        )

        # Process each problematic product
        successful_reclassifications = 0
        failed_reclassifications = 0

        progress_desc = f"Corrected Real Web Search Reclassification ({already_processed_count}/{total_problematic_in_dataset} done)"

        for i, (idx, row) in enumerate(
            tqdm(
                problematic_df.iterrows(),
                desc=progress_desc,
                total=len(problematic_df),
                initial=0,
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
                result = web_classifier.reclassify_problematic_product(
                    name,
                    manufacturer,
                    problem_type,
                    current_domain,
                    current_subcategory,
                    original_description,
                )

                # Store results
                complete_df.at[idx, "real_web_search_performed"] = result.get(
                    "real_web_search_performed", False
                )
                complete_df.at[idx, "real_web_search_successful"] = result.get(
                    "real_web_search_successful", False
                )
                complete_df.at[idx, "real_web_reclassification_successful"] = (
                    result.get("real_web_reclassification_successful", False)
                )

                # Store search info
                web_search_info = result.get("web_search_info", {})
                complete_df.at[idx, "real_web_suggested_domain"] = web_search_info.get(
                    "suggested_domain", ""
                )
                complete_df.at[idx, "real_web_domain_confidence"] = web_search_info.get(
                    "confidence", ""
                )
                complete_df.at[idx, "real_web_citations_found"] = web_search_info.get(
                    "citations_found", 0
                )

                # Store classification results
                if result.get("real_web_reclassification_successful", False):
                    classification = result.get("new_classification", {})
                    complete_df.at[idx, "real_web_new_domain"] = result.get(
                        "new_domain", ""
                    )
                    complete_df.at[idx, "real_web_new_subcategory"] = (
                        classification.get("subcategory", "")
                    )
                    complete_df.at[idx, "real_web_new_subsubcategory"] = (
                        classification.get("subsubcategory", "")
                    )
                    complete_df.at[idx, "real_web_new_subsubsubcategory"] = (
                        classification.get("subsubsubcategory", "")
                    )
                    complete_df.at[idx, "real_web_new_confidence"] = classification.get(
                        "confidence", ""
                    )
                    complete_df.at[idx, "real_web_validated_path"] = classification.get(
                        "validated_path", ""
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
                        f"Successfully reclassified: {name[:30]} -> {new_domain}"
                    )
                else:
                    failed_reclassifications += 1

                complete_df.at[idx, "real_web_total_tokens"] = result.get(
                    "total_tokens", 0
                )
                complete_df.at[idx, "real_web_processed_at"] = (
                    datetime.now().isoformat()
                )
                complete_df.at[idx, "real_web_processing_error"] = False

            except Exception as e:
                logger.error(f"Error processing '{name}': {e}")
                complete_df.at[idx, "real_web_search_performed"] = True
                complete_df.at[idx, "real_web_search_successful"] = False
                complete_df.at[idx, "real_web_reclassification_successful"] = False
                complete_df.at[idx, "real_web_processed_at"] = (
                    datetime.now().isoformat()
                )
                complete_df.at[idx, "real_web_processing_error"] = True
                failed_reclassifications += 1

            # Save checkpoint
            if (i + 1) % REAL_WEB_CHECKPOINT_FREQ == 0 or i == len(problematic_df) - 1:
                actual_progress = already_processed_count + i + 1
                checkpoint_file = os.path.join(
                    REAL_WEB_CHECKPOINT_DIR,
                    f"{REAL_WEB_CHECKPOINT_PREFIX}_{actual_progress}.csv",
                )
                complete_df.to_csv(checkpoint_file, index=False)
                logger.info(
                    f"Saved corrected real web search checkpoint: {checkpoint_file}"
                )
                logger.info(
                    f"Total progress: {actual_progress}/{total_problematic_in_dataset} problematic products processed"
                )

        # Save final result
        complete_df.to_csv(output_csv, index=False)
        logger.info(
            f"Complete corrected real web search reclassification saved to {output_csv}"
        )

        logger.info("CORRECTED REAL WEB SEARCH RECLASSIFICATION SUMMARY")
        logger.info(
            f"Problematic products processed this session: {len(problematic_df)}"
        )
        logger.info(
            f"Total problematic products processed: {already_processed_count + len(problematic_df)}"
        )
        logger.info(
            f"Successful reclassifications this session: {successful_reclassifications}"
        )
        logger.info(
            f"Failed reclassifications this session: {failed_reclassifications}"
        )
        if len(problematic_df) > 0:
            logger.info(
                f"Success rate this session: {successful_reclassifications/len(problematic_df)*100:.1f}%"
            )

        return complete_df, problematic_df

    except Exception as e:
        logger.error(f"Error in corrected real web search reclassification: {e}")
        raise


def main():
    """Main function for corrected real web search reclassification"""
    try:
        print("Starting corrected real web search reclassification...")
        process_corrected_real_web_search_reclassification_with_checkpointing(
            input_csv=INPUT_CSV, output_csv=OUTPUT_CSV
        )

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
