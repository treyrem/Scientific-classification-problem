# SYSTEMATIC ENHANCED YAML-BASED LLM-DRIVEN CLASSIFICATION SYSTEM
# Comprehensive YAML context with systematic product type analysis

import pandas as pd
import json
import logging
import os
import re
import yaml
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import random

# Configuration
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_llm_driven_classification1.csv"
VALIDATION_CSV = (
    "validation_llm_driven_classification_unlimited_claude_changes_prefilter.csv"
)
YAML_DIRECTORY = "C:/LabGit/150citations classification/category_structures"
MASTER_CATEGORIES_FILE = (
    "C:/LabGit/150citations classification/master_categories_claude.yaml"
)

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


class SystematicProductClassifier:
    """
    Systematic approach to product classification based on product type analysis
    rather than keyword matching or cherry-picked fixes
    """

    def __init__(self):
        self.product_type_framework = {
            "physical_instruments": {
                "indicators": [
                    "system",
                    "instrument",
                    "machine",
                    "analyzer",
                    "reader",
                    "scanner",
                    "microscope",
                    "centrifuge",
                    "thermocycler",
                    "spectrometer",
                ],
                "exclusions": [
                    "kit",
                    "reagent",
                    "buffer",
                    "solution",
                    "medium",
                    "stain",
                    "dye",
                    "membrane",
                ],
                "primary_domain": "Lab_Equipment",
                "reasoning": "Physical devices with electronics/mechanics",
            },
            "complete_assay_systems": {
                "indicators": [
                    "elisa kit",
                    "assay kit",
                    "detection kit",
                    "screening kit",
                    "immunoassay kit",
                    "multiplex kit",
                ],
                "exclusions": [
                    "buffer",
                    "individual reagent",
                    "single component",
                    "extraction",
                    "purification",
                ],
                "primary_domain": "Assay_Kits",
                "reasoning": "Complete ready-to-use measurement systems",
            },
            "sample_preparation_kits": {
                "indicators": [
                    "extraction kit",
                    "purification kit",
                    "isolation kit",
                    "cleanup kit",
                    "prep kit",
                    "preparation kit",
                ],
                "exclusions": ["assay", "detection", "measurement", "elisa"],
                "domain_logic": {
                    "dna|rna|nucleic": "Nucleic_Acid_Purification",
                    "protein": "Protein",
                    "cell|fixation|permeabilization": "Cell_Biology",
                    "plasmid|cloning|transformation": "Cloning_And_Expression",
                },
                "reasoning": "Kits for sample preparation, not detection",
            },
            "individual_reagents": {
                "indicators": [
                    "buffer",
                    "solution",
                    "reagent",
                    "medium",
                    "serum",
                    "stain",
                    "dye",
                ],
                "exclusions": ["kit", "system", "complete"],
                "domain_logic": {
                    "ripa|lysis|extraction": "Protein",
                    "culture|medium|serum|fbs": "Cell_Biology",
                    "acetonitrile|methanol|solvent|dmso": "Molecular_Biology",
                    "antibody|immunoglobulin": "Antibodies",
                    "giemsa|crystal violet|trypan blue": "Cell_Biology",
                },
                "reasoning": "Individual components, not complete systems",
            },
            "biological_materials": {
                "indicators": [
                    "cells",
                    "serum",
                    "plasma",
                    "tissue",
                    "membrane",
                    "antibody",
                ],
                "exclusions": ["synthetic", "kit system"],
                "domain_logic": {
                    "blood|serum|plasma": "Blood",
                    "cells|cell line": "Cell_Biology",
                    "membrane|pvdf|nitrocellulose|immobilon": "Protein",
                    "antibody|immunoglobulin": "Antibodies",
                },
                "reasoning": "Biological specimens and materials",
            },
        }

    def analyze_product_type(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """
        Systematically analyze product type using framework rules
        instead of cherry-picked keyword matching
        """

        text = f"{product_name} {description}".lower()

        analysis = {
            "product_name": product_name,
            "identified_type": None,
            "confidence": 0,
            "reasoning": "",
            "suggested_domain": None,
            "classification_logic": [],
        }

        # Score each product type systematically
        type_scores = {}

        for product_type, rules in self.product_type_framework.items():
            score = self._calculate_type_score(text, rules)
            type_scores[product_type] = score

            if score > 0:
                analysis["classification_logic"].append(
                    {
                        "type": product_type,
                        "score": score,
                        "matched_indicators": self._get_matched_indicators(
                            text, rules["indicators"]
                        ),
                        "exclusion_violations": self._get_exclusion_violations(
                            text, rules.get("exclusions", [])
                        ),
                    }
                )

        # Select highest scoring type
        if type_scores:
            best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            best_score = type_scores[best_type]

            if best_score > 0:
                analysis["identified_type"] = best_type
                analysis["confidence"] = min(best_score, 100)
                analysis["reasoning"] = self.product_type_framework[best_type][
                    "reasoning"
                ]
                analysis["suggested_domain"] = self._determine_domain(text, best_type)

        return analysis

    def _calculate_type_score(self, text: str, rules: Dict) -> float:
        """Calculate systematic score for product type match"""

        score = 0

        # Positive indicators
        indicators = rules.get("indicators", [])
        matched_indicators = 0
        for indicator in indicators:
            if indicator in text:
                matched_indicators += 1
                # Weight by specificity - longer phrases get higher scores
                score += len(indicator.split()) * 20

        # Exclusion penalties
        exclusions = rules.get("exclusions", [])
        for exclusion in exclusions:
            if exclusion in text:
                score -= 30  # Strong penalty for exclusions

        # Bonus for multiple indicator matches
        if matched_indicators > 1:
            score += matched_indicators * 10

        return max(0, score)  # Never negative

    def _get_matched_indicators(self, text: str, indicators: List[str]) -> List[str]:
        """Get list of matched indicators for transparency"""
        return [ind for ind in indicators if ind in text]

    def _get_exclusion_violations(self, text: str, exclusions: List[str]) -> List[str]:
        """Get list of exclusion violations for transparency"""
        return [exc for exc in exclusions if exc in text]

    def _determine_domain(self, text: str, product_type: str) -> str:
        """Systematically determine domain based on product type and content"""

        type_rules = self.product_type_framework[product_type]

        # For types with fixed domains
        if "primary_domain" in type_rules:
            return type_rules["primary_domain"]

        # For types with domain logic
        if "domain_logic" in type_rules:
            domain_logic = type_rules["domain_logic"]

            for pattern, domain in domain_logic.items():
                # Support regex-like patterns with |
                if "|" in pattern:
                    keywords = pattern.split("|")
                    if any(keyword in text for keyword in keywords):
                        return domain
                else:
                    if pattern in text:
                        return domain

        return "Other"  # Fallback


class SystematicPromptGenerator:
    """
    Generates systematic prompts based on product type analysis
    rather than hardcoded examples
    """

    def __init__(self, product_classifier: SystematicProductClassifier):
        self.classifier = product_classifier

    def generate_domain_selection_prompt(
        self,
        product_name: str,
        description: str,
        available_domains: List[str],
        domain_info_text: str,
    ) -> str:
        """Generate systematic domain selection prompt based on product type analysis"""

        # Analyze product type systematically
        type_analysis = self.classifier.analyze_product_type(product_name, description)

        # Generate systematic classification guidance
        systematic_guidance = self._generate_systematic_guidance(type_analysis)

        prompt = f"""You are a life science product classification expert using systematic product type analysis.

Product Name: "{product_name}"
Description: "{description}"

AVAILABLE DOMAINS ({len(available_domains)} total):
{domain_info_text}

SYSTEMATIC CLASSIFICATION FRAMEWORK:
{systematic_guidance}

SYSTEMATIC DECISION PROCESS:
1. IDENTIFY PRODUCT TYPE: What IS this product fundamentally?
   - Physical instrument with electronics/mechanics → Lab_Equipment
   - Complete ready-to-use assay system → Assay_Kits  
   - Sample preparation kit → Domain based on sample type
   - Individual reagent/buffer → Domain based on chemical function
   - Biological material → Domain based on biological source

2. APPLY DOMAIN LOGIC: Based on identified type, apply systematic rules
3. VERIFY CONSISTENCY: Ensure classification aligns with product function

SYSTEMATIC REASONING PRINCIPLES:
- Function over keywords: What does this product DO?
- Type over application: What IS it before considering WHERE it's used?
- System over components: Complete systems vs individual parts
- Specificity over generality: Use most specific applicable domain

SYSTEMATIC CLASSIFICATION EXAMPLES:
✅ "RIPA buffer" → Individual reagent → Protein (extraction buffer function)
✅ "fixation permeabilization kit" → Sample prep kit → Cell_Biology (cell preparation)
✅ "ELISA kit" → Complete assay system → Assay_Kits (detection system)
✅ "PCR machine" → Physical instrument → Lab_Equipment (measurement device)

Respond with JSON only:
{{
    "selected_domain": "exact_domain_name_with_spaces",
    "confidence": "High/Medium/Low",
    "product_type_identified": "{type_analysis.get('identified_type', 'unclear')}",
    "reasoning": "systematic explanation based on product type analysis focusing on WHAT the product IS"
}}"""

        return prompt

    def _generate_systematic_guidance(self, type_analysis: Dict) -> str:
        """Generate systematic guidance based on product type analysis"""

        if not type_analysis["identified_type"]:
            return (
                "ANALYSIS: Product type unclear - use general classification principles"
            )

        product_type = type_analysis["identified_type"]
        confidence = type_analysis["confidence"]
        reasoning = type_analysis["reasoning"]
        suggested_domain = type_analysis["suggested_domain"]

        guidance = f"""
SYSTEMATIC PRODUCT TYPE ANALYSIS:
- Identified Type: {product_type.replace('_', ' ').title()}
- Confidence: {confidence}%
- Reasoning: {reasoning}
- Suggested Domain: {suggested_domain}

CLASSIFICATION LOGIC APPLIED:"""

        for logic in type_analysis["classification_logic"]:
            if logic["score"] > 0:
                guidance += f"""
- {logic['type'].replace('_', ' ').title()}: Score {logic['score']}
  * Matched indicators: {logic['matched_indicators']}
  * Exclusion violations: {logic['exclusion_violations']}"""

        return guidance

    def generate_within_domain_prompt(
        self,
        product_name: str,
        description: str,
        domain_key: str,
        structure: str,
        type_analysis: Dict,
    ) -> str:
        """Generate systematic within-domain classification prompt"""

        prompt = f"""You are classifying within the {domain_key} domain using systematic principles.

Product: "{product_name}"
Description: "{description}"

DOMAIN STRUCTURE:
{structure}

SYSTEMATIC PRODUCT TYPE ANALYSIS:
{self._format_type_analysis_for_prompt(type_analysis)}

SYSTEMATIC CLASSIFICATION PRINCIPLES:
1. Function-based classification: Navigate to subcategories based on what the product DOES
2. Specificity principle: Use the most specific applicable path
3. Consistency check: Ensure path aligns with product type and function

SYSTEMATIC VALIDATION RULES:
- Individual reagents (buffers, solutions) → Reagent categories, NOT assay categories
- Complete kits → Kit categories, NOT individual component categories  
- Physical instruments → Equipment categories, NOT reagent categories
- Sample prep tools → Preparation categories, NOT detection categories

Use this systematic analysis to navigate to the most appropriate and specific classification path.

Respond with JSON only:
{{
    "subcategory": "exact_name",
    "subsubcategory": "exact_name_or_null", 
    "subsubsubcategory": "exact_name_or_null",
    "reasoning": "systematic explanation based on function and type analysis",
    "confidence": "High/Medium/Low"
}}"""

        return prompt

    def _format_type_analysis_for_prompt(self, analysis: Dict) -> str:
        """Format type analysis for inclusion in prompts"""

        if not analysis.get("identified_type"):
            return "Product type: Unclear - apply general principles"

        return f"""
Product Type: {analysis['identified_type'].replace('_', ' ').title()}
Confidence: {analysis['confidence']}%
Reasoning: {analysis['reasoning']}
Suggested Domain: {analysis['suggested_domain']}
"""


class LLMDrivenCategorySystem:
    """Enhanced LLM-driven category system with systematic classification"""

    def __init__(self, master_file: str = None, yaml_directory: str = None):
        self.master_file = master_file or MASTER_CATEGORIES_FILE
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.master_config = {}
        self.category_files = {}
        self.domain_structures = {}
        self.available_domains = []

        # Initialize systematic components
        self.product_classifier = SystematicProductClassifier()
        self.prompt_generator = SystematicPromptGenerator(self.product_classifier)

        self.load_master_categories()
        self.load_individual_yaml_files()

    def load_master_categories(self):
        """Load the master categories configuration"""
        try:
            with open(self.master_file, "r", encoding="utf-8") as f:
                self.master_config = yaml.safe_load(f)
            logger.info(f"✓ Loaded master categories from {self.master_file}")
        except Exception as e:
            logger.error(f"Failed to load master categories: {e}")
            raise

    def load_individual_yaml_files(self):
        """Load individual YAML files and extract their content"""
        domain_mapping = self.master_config.get("domain_mapping", {})

        for domain_key, domain_info in domain_mapping.items():
            yaml_file = domain_info.get("yaml_file")
            if not yaml_file:
                continue

            yaml_path = os.path.join(self.yaml_directory, yaml_file)

            try:
                if os.path.exists(yaml_path):
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)

                    structure = self._extract_structure(data)
                    self.category_files[domain_key] = structure
                    self.domain_structures[domain_key] = structure
                    self.available_domains.append(domain_key)
                    logger.info(f"✓ Loaded {domain_key} from {yaml_file}")
                else:
                    logger.warning(f"⚠ File not found: {yaml_file}")

            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")

        logger.info(f"Available domains: {self.available_domains}")

    def _extract_structure(self, data):
        """Extract structure from YAML data regardless of format"""
        if not data:
            return {}

        if "categories" in data:
            return data["categories"]
        elif "category" in data:
            return data["category"]
        else:
            return data

    def get_systematic_domain_selection_prompt(
        self, product_name: str, description: str
    ) -> str:
        """Generate systematic domain selection prompt"""

        domain_info = []
        domain_mapping = {}  # Map display names to actual keys

        for domain_key in self.available_domains:
            domain_data = self.master_config.get("domain_mapping", {}).get(
                domain_key, {}
            )
            domain_desc = domain_data.get("description", "")
            typical_products = domain_data.get("typical_products", [])
            keywords = domain_data.get("keywords", [])

            # Create display name and map it to the actual key
            display_name = domain_key.replace("_", " ")
            domain_mapping[display_name] = domain_key

            # Show keywords for better matching
            key_keywords = ", ".join(keywords[:6]) if keywords else "N/A"

            domain_info.append(
                f"""
{display_name}:
  Description: {domain_desc}
  Key indicators: {key_keywords}
  Examples: {', '.join(typical_products[:3]) if typical_products else 'N/A'}"""
            )

        domains_text = "\n".join(domain_info)

        # Store the mapping for later use
        self._domain_mapping = domain_mapping

        # Use systematic prompt generator
        return self.prompt_generator.generate_domain_selection_prompt(
            product_name, description, self.available_domains, domains_text
        )

    def select_domain_systematically(
        self, product_name: str, description: str
    ) -> Dict[str, Any]:
        """Use systematic approach for domain selection"""

        prompt = self.get_systematic_domain_selection_prompt(product_name, description)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)
            result = json.loads(cleaned)

            selected_domain_display = result.get("selected_domain", "Other")

            # Convert display name back to domain key using the mapping
            domain_mapping = getattr(self, "_domain_mapping", {})
            selected_domain = domain_mapping.get(
                selected_domain_display, selected_domain_display
            )

            # If still not found, try direct lookup or fuzzy matching
            if (
                selected_domain not in self.available_domains
                and selected_domain != "Other"
            ):
                # Try direct underscore conversion
                underscore_version = selected_domain_display.replace(" ", "_")
                if underscore_version in self.available_domains:
                    selected_domain = underscore_version
                else:
                    # Try fuzzy matching
                    for domain_key in self.available_domains:
                        if (
                            domain_key.replace("_", " ").lower()
                            == selected_domain_display.lower()
                        ):
                            selected_domain = domain_key
                            break
                    else:
                        logger.warning(
                            f"LLM selected unrecognized domain '{selected_domain_display}', using Other"
                        )
                        selected_domain = "Other"
                        result["confidence"] = "Low"
                        result["reasoning"] = (
                            f"Unrecognized domain selected: {selected_domain_display}"
                        )

            result["selected_domain"] = selected_domain
            result["original_selection"] = selected_domain_display
            result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            # Add systematic analysis
            type_analysis = self.product_classifier.analyze_product_type(
                product_name, description
            )
            result["systematic_analysis"] = type_analysis

            logger.info(
                f"Systematic domain selection for '{product_name}': {selected_domain} (type: {type_analysis.get('identified_type', 'unclear')})"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error in systematic domain selection for '{product_name}': {e}"
            )
            return {
                "selected_domain": "Other",
                "confidence": "Low",
                "reasoning": f"Error in domain selection: {e}",
                "token_usage": 0,
                "systematic_analysis": {"identified_type": None, "confidence": 0},
            }

    def get_focused_structure_for_prompt(
        self, domain_key: str, product_name: str = "", description: str = ""
    ) -> str:
        """Enhanced structure prompt with comprehensive YAML context and depth guidance"""

        if domain_key not in self.domain_structures:
            return f"Domain '{domain_key}' not found"

        # Get enhanced master category info
        domain_info = self.master_config.get("domain_mapping", {}).get(domain_key, {})

        lines = [f"DOMAIN: {domain_key.replace('_', ' ')}"]
        lines.append(f"Description: {domain_info.get('description', '')}")
        lines.append("")

        # Show COMPLETE deep paths from the YAML structure (unlimited context)
        complete_paths = self._extract_actual_paths_from_yaml(domain_key)
        if complete_paths:
            lines.append(
                f"COMPREHENSIVE CLASSIFICATION PATHS ({len(complete_paths)} total paths - use these exact names for deep classification):"
            )

            # Group paths by depth for clarity
            depth_2_paths = [p for p in complete_paths if len(p.split(" -> ")) == 2]
            depth_3_paths = [p for p in complete_paths if len(p.split(" -> ")) == 3]
            depth_4_plus_paths = [
                p for p in complete_paths if len(p.split(" -> ")) >= 4
            ]

            if depth_2_paths:
                lines.append("\n2-LEVEL PATHS (subcategory -> subsubcategory):")
                for path in depth_2_paths[:8]:
                    lines.append(f"- {path}")

            if depth_3_paths:
                lines.append(
                    "\n3-LEVEL PATHS (subcategory -> subsubcategory -> subsubsubcategory):"
                )
                for path in depth_3_paths[:10]:
                    lines.append(f"- {path}")

            if depth_4_plus_paths:
                lines.append("\n4+ LEVEL PATHS (deepest available):")
                for path in depth_4_plus_paths[:8]:
                    lines.append(f"- {path}")

            lines.append("")

        # Show master category paths as guidance
        key_paths = domain_info.get("key_classification_paths", [])
        if key_paths:
            lines.append("CLASSIFICATION GUIDANCE:")
            for path in key_paths[:8]:
                lines.append(f"- {path}")
            lines.append("")

        # Show classification hints
        hints = domain_info.get("classification_hints", [])
        if hints:
            lines.append("CLASSIFICATION HINTS:")
            for hint in hints[:5]:
                lines.append(f"- {hint}")
            lines.append("")

        # Enhanced depth requirements for comprehensive context
        lines.append("ENHANCED DEPTH REQUIREMENTS:")
        lines.append("1. MANDATORY: Provide at least subcategory (level 1)")
        lines.append("2. STRONGLY RECOMMENDED: Provide subsubcategory (level 2)")
        lines.append("3. PREFERRED: Provide subsubsubcategory (level 3) when available")
        lines.append("4. OPTIMAL: Use 4+ levels when comprehensive paths support it")
        lines.append(
            "5. Use EXACT names from the comprehensive classification paths shown above"
        )
        lines.append("6. Navigate as deep as the available structure allows")

        return "\n".join(lines)

    def _extract_actual_paths_from_yaml(self, domain_key: str) -> List[str]:
        """Extract complete deep classification paths from YAML structure with UNLIMITED context"""
        if domain_key not in self.domain_structures:
            return []

        structure = self.domain_structures[domain_key]
        complete_paths = []

        def extract_complete_paths(node, current_path=[], max_depth=4):
            if len(current_path) >= max_depth:
                return

            if isinstance(node, dict):
                # Handle top-level domain structure (e.g., {"Antibodies": {...}})
                if len(current_path) == 0 and len(node) == 1:
                    domain_name = list(node.keys())[0]
                    extract_complete_paths(node[domain_name], [domain_name])
                    return

                # Handle subcategories navigation
                if "subcategories" in node:
                    extract_complete_paths(node["subcategories"], current_path)
                    return

                # Handle subsubcategories navigation
                if "subsubcategories" in node:
                    extract_complete_paths(node["subsubcategories"], current_path)
                    return

                # Regular structure navigation - build complete paths
                for key, value in node.items():
                    new_path = current_path + [key]

                    # Add complete paths at different depths
                    if len(new_path) >= 2:  # At least domain + subcategory
                        # Show complete path without domain prefix
                        display_path = " -> ".join(new_path[1:])
                        complete_paths.append(display_path)

                    # Continue deeper to find more complete paths
                    if value and isinstance(value, (dict, list)):
                        extract_complete_paths(value, new_path)

                # Handle final level lists within dictionaries
                for key, value in node.items():
                    if isinstance(value, list) and value:
                        new_path = current_path + [key]
                        for item in value:
                            if isinstance(item, str):
                                final_path = new_path + [item]
                                if len(final_path) >= 2:
                                    display_path = " -> ".join(final_path[1:])
                                    complete_paths.append(display_path)

            elif isinstance(node, list):
                # Handle list of final items
                for item in node:
                    if isinstance(item, str):
                        new_path = current_path + [item]
                        if len(new_path) >= 2:
                            display_path = " -> ".join(new_path[1:])
                            complete_paths.append(display_path)
                    elif isinstance(item, dict):
                        extract_complete_paths(item, current_path)

        extract_complete_paths(structure)

        # Sort by depth (deeper paths first) and remove duplicates
        unique_paths = list(set(complete_paths))

        # Sort by depth first (longer paths first), then alphabetically
        unique_paths.sort(key=lambda x: (-len(x.split(" -> ")), x))

        # Return ALL paths for comprehensive context (unlimited)
        return unique_paths

    def validate_classification_path(
        self, domain_key: str, path_components: List[str]
    ) -> Tuple[bool, str]:
        """Enhanced validation with better fuzzy matching and fallback"""
        if domain_key not in self.domain_structures:
            return False, f"Domain '{domain_key}' not found"

        structure = self.domain_structures[domain_key]
        validated_path = [domain_key.replace("_", " ")]

        # Navigate into the domain structure
        current_node = structure

        # Handle top-level domain structure (e.g., {"Lab Equipment": {...}})
        if isinstance(current_node, dict) and len(current_node) == 1:
            domain_name = list(current_node.keys())[0]
            current_node = current_node[domain_name]
            validated_path = [domain_name]  # Use actual domain name from YAML

        logger.info(
            f"Validating path components: {path_components} in domain {domain_key}"
        )

        # If no path components, accept at domain level
        if not path_components or all(
            not comp or comp == "null" for comp in path_components
        ):
            final_path = " -> ".join(validated_path)
            logger.info(f"Accepting domain-level classification: {final_path}")
            return True, final_path

        # Navigate through the structure with enhanced fuzzy matching
        for i, component in enumerate(path_components):
            if not component or component == "null":
                continue

            found = False

            # Navigate into subcategories if this is the first level
            if (
                isinstance(current_node, dict)
                and "subcategories" in current_node
                and i == 0
            ):
                current_node = current_node["subcategories"]
                logger.info(f"Navigated into subcategories level")

            if isinstance(current_node, dict):
                # Enhanced matching strategies
                component_lower = component.lower()

                # Strategy 1: Exact match
                if component in current_node:
                    validated_path.append(component)
                    current_node = current_node[component]
                    found = True
                    logger.info(f"Exact match: '{component}'")

                # Strategy 2: Case-insensitive exact match
                elif not found:
                    for key in current_node.keys():
                        if key.lower() == component_lower:
                            validated_path.append(key)
                            current_node = current_node[key]
                            found = True
                            logger.info(
                                f"Case-insensitive match: '{component}' -> '{key}'"
                            )
                            break

                # Strategy 3: Partial matching (both ways)
                elif not found:
                    matches = []
                    for key in current_node.keys():
                        key_lower = key.lower()
                        # Check if component is contained in key OR key is contained in component
                        if component_lower in key_lower or key_lower in component_lower:
                            matches.append(
                                (
                                    key,
                                    max(len(component_lower), len(key_lower))
                                    - abs(len(component_lower) - len(key_lower)),
                                )
                            )

                    if matches:
                        # Sort by best match (longest common part)
                        best_match = max(matches, key=lambda x: x[1])[0]
                        validated_path.append(best_match)
                        current_node = current_node[best_match]
                        found = True
                        logger.info(f"Partial match: '{component}' -> '{best_match}'")

                # Navigate deeper if we found a match
                if found and isinstance(current_node, dict):
                    if "subsubcategories" in current_node:
                        current_node = current_node["subsubcategories"]
                        logger.info(f"Navigated into subsubcategories level")

            elif isinstance(current_node, list):
                # Enhanced list matching
                component_lower = component.lower()

                # Strategy 1: Exact match
                exact_matches = [
                    item
                    for item in current_node
                    if isinstance(item, str) and item == component
                ]
                if exact_matches:
                    validated_path.append(exact_matches[0])
                    found = True
                    logger.info(f"Exact match in list: '{exact_matches[0]}'")

                # Strategy 2: Case-insensitive match
                elif not found:
                    case_matches = [
                        item
                        for item in current_node
                        if isinstance(item, str) and item.lower() == component_lower
                    ]
                    if case_matches:
                        validated_path.append(case_matches[0])
                        found = True
                        logger.info(
                            f"Case-insensitive match in list: '{case_matches[0]}'"
                        )

                # Strategy 3: Partial matches
                elif not found:
                    partial_matches = [
                        item
                        for item in current_node
                        if isinstance(item, str)
                        and (
                            component_lower in item.lower()
                            or item.lower() in component_lower
                        )
                    ]
                    if partial_matches:
                        # Choose the best match (closest length)
                        best_match = min(
                            partial_matches, key=lambda x: abs(len(x) - len(component))
                        )
                        validated_path.append(best_match)
                        found = True
                        logger.info(
                            f"Partial match in list: '{component}' -> '{best_match}'"
                        )

            if not found:
                # More lenient acceptance - if we got at least domain + one level, accept it
                if len(validated_path) >= 2:  # Domain + at least one subcategory
                    logger.info(
                        f"Stopping validation at: {' -> '.join(validated_path)} (couldn't find '{component}', but accepting partial path)"
                    )
                    break
                else:
                    # Accept at domain level if nothing else works
                    logger.info(
                        f"No matches found for '{component}', accepting at domain level"
                    )
                    break

        # Always return True with best effort path
        final_path = " -> ".join(validated_path)
        logger.info(f"Successfully validated path: {final_path}")
        return True, final_path

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


class LLMDrivenClassifier:
    """Enhanced LLM-driven classifier with systematic classification support"""

    def __init__(self, category_system: LLMDrivenCategorySystem):
        self.category_system = category_system
        self.logger = logging.getLogger(__name__)

    @property
    def available_domains(self):
        """Property to access available domains from category system"""
        return self.category_system.available_domains

    @property
    def master_config(self):
        """Property to access master config from category system"""
        return self.category_system.master_config

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()

    def classify_product(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Enhanced classification with systematic approach"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SYSTEMATIC CLASSIFYING: '{product_name}'")
        self.logger.info(f"{'='*60}")

        # Step 1: Get domain using systematic approach
        domain_result = self.category_system.select_domain_systematically(
            product_name, description
        )

        total_tokens = domain_result.get("token_usage", 0)
        selected_domain = domain_result.get("selected_domain", "Other")

        # Step 2: Classify within domain using systematic approach
        if selected_domain == "Other":
            return self._create_fallback_result(
                product_name,
                {"reasoning": "No suitable domain found", "token_usage": total_tokens},
            )

        classification = self._classify_within_domain_systematically(
            product_name, description, selected_domain
        )

        total_tokens += classification.get("token_usage", 0) if classification else 0

        if classification:
            classification.update(
                {
                    "is_primary": True,
                    "domain_confidence": 95,
                    "domain_reasoning": f"Systematic domain match: {selected_domain}",
                    "systematic_analysis": domain_result.get("systematic_analysis", {}),
                }
            )
            return self._format_single_classification(classification, total_tokens)
        else:
            return self._create_fallback_result(
                product_name,
                {
                    "reasoning": "Domain classification failed",
                    "token_usage": total_tokens,
                },
            )

    def _format_single_classification(
        self, classification: Dict, total_tokens: int
    ) -> Dict[str, Any]:
        """Format single classification result"""
        return {
            "classifications": [classification],
            "primary_classification": classification,
            "secondary_classification": None,
            "is_dual_function": False,
            "dual_function_pattern": "",
            "dual_function_reasoning": "",
            "classification_count": 1,
            "total_token_usage": total_tokens,
        }

    def _classify_within_domain_systematically(
        self, product_name: str, description: str, domain_key: str
    ) -> Optional[Dict]:
        """Systematic within-domain classification"""

        # Get focused structure and guidance with comprehensive YAML context
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        # Get systematic product type analysis
        type_analysis = self.category_system.product_classifier.analyze_product_type(
            product_name, description
        )

        # Generate systematic prompt
        prompt = self.category_system.prompt_generator.generate_within_domain_prompt(
            product_name, description, domain_key, focused_structure, type_analysis
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self.category_system._strip_code_fences(text)

            self.logger.info(
                f"Systematic LLM Response for {product_name}: {cleaned[:200]}..."
            )

            result = json.loads(cleaned)

            # Calculate actual depth achieved
            depth_achieved = 0
            levels = ["subcategory", "subsubcategory", "subsubsubcategory"]
            for level in levels:
                value = result.get(level, "")
                if value and value != "null" and value.strip():
                    depth_achieved += 1
                else:
                    break

            result["depth_achieved"] = depth_achieved

            # Enhanced logging for systematic context
            self.logger.info(
                f"SYSTEMATIC Classification depth achieved for '{product_name}': {depth_achieved} levels"
            )
            self.logger.info(
                f"Product type identified: {type_analysis.get('identified_type', 'unclear')}"
            )

            # Validate the classification path
            path_components = [
                result.get("subcategory", ""),
                result.get("subsubcategory", ""),
                result.get("subsubsubcategory", ""),
            ]
            path_components = [
                comp for comp in path_components if comp and comp != "null"
            ]

            self.logger.info(
                f"Attempting to validate path: {path_components} in domain {domain_key}"
            )

            is_valid, validated_path = (
                self.category_system.validate_classification_path(
                    domain_key, path_components
                )
            )

            # Apply systematic validation
            result = self._validate_systematically(
                result, product_name, domain_key, type_analysis
            )

            result.update(
                {
                    "domain": domain_key,
                    "is_valid_path": is_valid,
                    "validated_path": validated_path,
                    "is_primary": True,
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                    "systematic_type_analysis": type_analysis,
                }
            )

            self.logger.info(
                f"Systematic analysis result: type={type_analysis.get('identified_type', 'unclear')}, valid_path={is_valid}, depth={depth_achieved}"
            )

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error for '{product_name}': {cleaned}")
            self.logger.error(f"Error details: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in systematic domain analysis {domain_key}: {e}")
            return None

    def _validate_systematically(
        self, result: Dict, product_name: str, domain_key: str, type_analysis: Dict
    ) -> Dict:
        """Validate classification using systematic principles, not cherry-picked fixes"""

        validation_issues = []

        # Rule 1: Individual reagents shouldn't be in assay subcategories
        if (
            type_analysis.get("identified_type") == "individual_reagents"
            and "assay" in result.get("subsubcategory", "").lower()
        ):
            validation_issues.append("individual_reagent_in_assay_category")

            # Systematic fix: Move to appropriate reagent category
            if domain_key == "Protein":
                result["subcategory"] = "Biochemistry Reagents"
                result["subsubcategory"] = "Buffers"
                result["reasoning"] = (
                    "SYSTEMATIC FIX: Individual reagent moved to reagent category"
                )
            elif domain_key == "Molecular_Biology":
                result["subcategory"] = "Chemicals"
                result["subsubcategory"] = "Solvents"
                result["reasoning"] = (
                    "SYSTEMATIC FIX: Individual chemical moved to chemical category"
                )

        # Rule 2: Complete systems shouldn't be in individual component categories
        if (
            type_analysis.get("identified_type") == "complete_assay_systems"
            and result.get("subcategory", "").lower().find("individual") != -1
        ):
            validation_issues.append("complete_system_in_component_category")

            # Systematic fix: Move to system category
            if domain_key == "Assay_Kits":
                result["subcategory"] = "Cell-Based Assays"
                result["reasoning"] = (
                    "SYSTEMATIC FIX: Complete system moved to system category"
                )

        # Rule 3: Sample prep kits should be in preparation categories
        if (
            type_analysis.get("identified_type") == "sample_preparation_kits"
            and "assay" in result.get("subcategory", "").lower()
        ):
            validation_issues.append("sample_prep_kit_in_assay_category")

            # Systematic fix based on domain
            if domain_key == "Cell_Biology":
                result["subcategory"] = "Cell Analysis"
                result["subsubcategory"] = "Flow Cytometry"
                result["reasoning"] = (
                    "SYSTEMATIC FIX: Sample prep kit moved to preparation category"
                )

        if validation_issues:
            result["systematic_validation_applied"] = validation_issues
            result["confidence"] = "Medium"  # Lower confidence for validated results
            self.logger.info(
                f"Applied systematic validation fixes for '{product_name}': {validation_issues}"
            )

        return result

    def _create_fallback_result(
        self, product_name: str, domain_selection: Dict
    ) -> Dict[str, Any]:
        """Create fallback result when classification fails"""

        self.logger.warning(f"Using fallback classification for: {product_name}")

        fallback_classification = {
            "domain": "Other",
            "subcategory": "Unclassified",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "domain_fit_score": 10,
            "reasoning": domain_selection.get(
                "reasoning", "Could not determine specific classification"
            ),
            "is_valid_path": False,
            "validated_path": "Other -> Unclassified",
            "is_primary": True,
            "domain_selection_confidence": domain_selection.get("confidence", "Low"),
            "domain_selection_reasoning": domain_selection.get("reasoning", ""),
            "token_usage": domain_selection.get("token_usage", 0),
        }

        return {
            "classifications": [fallback_classification],
            "primary_classification": fallback_classification,
            "secondary_classification": None,
            "classification_count": 1,
            "is_dual_function": False,
            "dual_function_pattern": "",
            "dual_function_reasoning": "",
            "total_token_usage": domain_selection.get("token_usage", 0),
        }


def test_systematic_classification():
    """Test the systematic classification system"""

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    # Test cases including the problematic ones
    test_cases = [
        {
            "name": "Anti-CD3 monoclonal antibody",
            "description": "Monoclonal antibody against CD3 for flow cytometry applications",
            "expected_domain": "Antibodies",
        },
        {
            "name": "qPCR master mix",
            "description": "Ready-to-use mix for quantitative PCR amplification",
            "expected_domain": "PCR",
        },
        {
            "name": "RNA extraction kit",
            "description": "Kit for total RNA isolation from cells and tissues",
            "expected_domain": "Nucleic_Acid_Purification",
        },
        {
            "name": "Cell culture medium",
            "description": "Complete medium for mammalian cell culture",
            "expected_domain": "Cell_Biology",
        },
        {
            "name": "Microplate reader",
            "description": "Automated plate reading system for absorbance and fluorescence",
            "expected_domain": "Lab_Equipment",
        },
        {
            "name": "giemsa",
            "description": "Staining reagent for microscopy",
            "expected_domain": "Cell_Biology",
        },
        {
            "name": "PVDF membrane",
            "description": "Membrane for Western blotting",
            "expected_domain": "Protein",
        },
        # Add the problematic cases
        {
            "name": "ZIBF fixation permeabilization kit",
            "description": "Kit for cell fixation and permeabilization",
            "expected_domain": "Cell_Biology",
        },
        {
            "name": "RIPA buffer",
            "description": "Radio immunoprecipitation assay buffer for protein extraction",
            "expected_domain": "Protein",
        },
    ]

    print("=" * 80)
    print("TESTING SYSTEMATIC CLASSIFICATION SYSTEM")
    print("=" * 80)

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTEST {i}: {test_case['name']}")
        print("-" * 50)

        # Full classification
        result = classifier.classify_product(
            test_case["name"], test_case["description"]
        )

        primary = result.get("primary_classification", {})
        selected_domain = primary.get("domain", "")
        expected_domain = test_case["expected_domain"]

        # Get systematic analysis info
        systematic_analysis = primary.get("systematic_type_analysis", {})
        product_type = systematic_analysis.get("identified_type", "unclear")

        print(f"Expected Domain: {expected_domain}")
        print(f"Selected Domain: {selected_domain}")
        print(f"Product Type Identified: {product_type}")

        domain_correct = selected_domain == expected_domain
        path_valid = primary.get("is_valid_path", False)
        depth_achieved = primary.get("depth_achieved", 0)
        validation_applied = primary.get("systematic_validation_applied", [])

        print(f"Domain Selection: {'✓ CORRECT' if domain_correct else '✗ INCORRECT'}")
        print(f"Path Valid: {'✓ YES' if path_valid else '✗ NO'}")
        print(f"Classification Depth: {depth_achieved} levels")
        print(f"Confidence: {primary.get('confidence', 'N/A')}")
        print(f"Validated Path: {primary.get('validated_path', 'N/A')}")
        print(f"Systematic Validation Applied: {validation_applied}")
        print(f"Reasoning: {primary.get('reasoning', 'N/A')}")
        print(f"Token Usage: {result.get('total_token_usage', 0)}")

        if domain_correct and path_valid:
            success_count += 1
            print("✅ OVERALL: SUCCESS")
        else:
            print("❌ OVERALL: FAILED")

    print(f"\n{'='*80}")
    print(
        f"SYSTEMATIC CLASSIFICATION TEST RESULTS: {success_count}/{len(test_cases)} passed ({success_count/len(test_cases)*100:.1f}%)"
    )
    print(f"{'='*80}")

    return success_count >= 7  # Expect high success rate with systematic approach


def process_validation_sample():
    """Process validation sample with systematic classification"""
    logger.info("Starting systematic validation sample processing...")

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        sample_size = min(100, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        validation_df = df.iloc[sample_indices].copy()

        # Add ALL classification columns including systematic analysis
        validation_df["primary_domain"] = ""
        validation_df["primary_subcategory"] = ""
        validation_df["primary_subsubcategory"] = ""
        validation_df["primary_subsubsubcategory"] = ""
        validation_df["primary_confidence"] = ""
        validation_df["primary_fit_score"] = 0
        validation_df["primary_path_valid"] = False
        validation_df["domain_selection_confidence"] = ""
        validation_df["domain_selection_reasoning"] = ""
        validation_df["validated_path"] = ""
        validation_df["total_token_usage"] = 0
        validation_df["systematic_product_type"] = ""
        validation_df["systematic_confidence"] = 0
        validation_df["systematic_validation_applied"] = ""

        # Add missing dual classification columns (keeping for compatibility)
        validation_df["secondary_domain"] = ""
        validation_df["secondary_subcategory"] = ""
        validation_df["secondary_subsubcategory"] = ""
        validation_df["secondary_subsubsubcategory"] = ""
        validation_df["secondary_confidence"] = ""
        validation_df["secondary_fit_score"] = 0
        validation_df["secondary_path_valid"] = False
        validation_df["secondary_validated_path"] = ""
        validation_df["is_dual_function"] = False
        validation_df["dual_function_pattern"] = ""
        validation_df["dual_function_reasoning"] = ""
        validation_df["classification_count"] = 1

        logger.info(
            f"Processing {len(validation_df)} products with systematic classification..."
        )

        for idx in tqdm(validation_df.index, desc="Systematic Classification"):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            # Perform systematic classification
            result = classifier.classify_product(name, description)

            # Store primary results
            primary = result.get("primary_classification", {})
            validation_df.at[idx, "primary_domain"] = primary.get("domain", "")
            validation_df.at[idx, "primary_subcategory"] = primary.get(
                "subcategory", ""
            )
            validation_df.at[idx, "primary_subsubcategory"] = primary.get(
                "subsubcategory", ""
            )
            validation_df.at[idx, "primary_subsubsubcategory"] = primary.get(
                "subsubsubcategory", ""
            )
            validation_df.at[idx, "primary_confidence"] = primary.get("confidence", "")
            validation_df.at[idx, "primary_path_valid"] = primary.get(
                "is_valid_path", False
            )
            validation_df.at[idx, "validated_path"] = primary.get("validated_path", "")
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )

            # Store systematic analysis results
            systematic_analysis = primary.get("systematic_type_analysis", {})
            validation_df.at[idx, "systematic_product_type"] = systematic_analysis.get(
                "identified_type", ""
            )
            validation_df.at[idx, "systematic_confidence"] = systematic_analysis.get(
                "confidence", 0
            )
            validation_df.at[idx, "systematic_validation_applied"] = str(
                primary.get("systematic_validation_applied", [])
            )

            # Store basic dual classification info (mostly will be single for systematic approach)
            validation_df.at[idx, "classification_count"] = result.get(
                "classification_count", 1
            )
            validation_df.at[idx, "is_dual_function"] = result.get(
                "is_dual_function", False
            )

        # Save results
        validation_df.to_csv(VALIDATION_CSV, index=False)
        logger.info(f"Systematic validation sample saved to {VALIDATION_CSV}")

        # Generate report
        generate_systematic_validation_report(validation_df)

        return validation_df

    except Exception as e:
        logger.error(f"Error in systematic validation processing: {e}")
        raise


def generate_systematic_validation_report(validation_df: pd.DataFrame):
    """Generate systematic validation report"""
    print("\n" + "=" * 80)
    print("SYSTEMATIC CLASSIFICATION VALIDATION REPORT")
    print("=" * 80)

    total_products = len(validation_df)
    classified_products = len(
        validation_df[
            validation_df["primary_domain"].notna()
            & (validation_df["primary_domain"] != "")
            & (validation_df["primary_domain"] != "Other")
        ]
    )

    print(f"Total products validated: {total_products}")
    print(
        f"Successfully classified: {classified_products} ({classified_products/total_products*100:.1f}%)"
    )

    # Systematic product type analysis
    print(f"\n{'SYSTEMATIC PRODUCT TYPE ANALYSIS':-^60}")
    type_counts = validation_df["systematic_product_type"].value_counts()
    for product_type, count in type_counts.items():
        if product_type:  # Skip empty values
            percentage = count / total_products * 100
            print(
                f"  {product_type.replace('_', ' ').title():<35} {count:>5} ({percentage:>5.1f}%)"
            )

    # Systematic validation fixes applied
    validation_fixes = validation_df[
        validation_df["systematic_validation_applied"] != "[]"
    ]
    if len(validation_fixes) > 0:
        print(f"\n{'SYSTEMATIC VALIDATION FIXES APPLIED':-^60}")
        print(
            f"  Products with systematic fixes: {len(validation_fixes)} ({len(validation_fixes)/total_products*100:.1f}%)"
        )

        # Show examples of fixes
        for idx, row in validation_fixes.head(3).iterrows():
            print(f"  {row['Name'][:40]:<40} → {row['systematic_validation_applied']}")

    # Enhanced depth analysis
    def calculate_depth(row):
        depth = 0
        levels = [
            "primary_subcategory",
            "primary_subsubcategory",
            "primary_subsubsubcategory",
        ]
        for level in levels:
            value = row.get(level, "")
            if value and str(value) != "nan" and str(value).strip():
                depth += 1
            else:
                break
        return depth

    validation_df["classification_depth"] = validation_df.apply(calculate_depth, axis=1)
    depth_counts = validation_df["classification_depth"].value_counts().sort_index()

    print(f"\n{'SYSTEMATIC CLASSIFICATION DEPTH ANALYSIS':-^60}")
    for depth, count in depth_counts.items():
        percentage = count / total_products * 100
        print(f"  Depth {depth}: {count:>5} ({percentage:>5.1f}%)")

    avg_depth = validation_df["classification_depth"].mean()
    print(f"  Average depth with systematic analysis: {avg_depth:.2f}")

    # Quality improvements
    depth_3_plus = len(validation_df[validation_df["classification_depth"] >= 3])
    print(
        f"  Deep classifications (3+ levels): {depth_3_plus} ({depth_3_plus/total_products*100:.1f}%)"
    )

    # Domain distribution
    print(f"\n{'DOMAIN DISTRIBUTION':-^60}")
    domain_counts = validation_df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(10).items():
        print(f"  {domain:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Path validity
    valid_paths = len(validation_df[validation_df["primary_path_valid"] == True])
    print(f"\n{'PATH VALIDITY':-^60}")
    print(f"  Valid paths: {valid_paths} ({valid_paths/total_products*100:.1f}%)")

    # Systematic confidence analysis
    avg_systematic_confidence = validation_df["systematic_confidence"].mean()
    print(f"\n{'SYSTEMATIC CONFIDENCE ANALYSIS':-^60}")
    print(f"  Average systematic confidence: {avg_systematic_confidence:.1f}%")

    # Token usage analysis
    total_tokens = validation_df["total_token_usage"].sum()
    avg_tokens = validation_df["total_token_usage"].mean()
    print(f"\n{'SYSTEMATIC TOKEN USAGE ANALYSIS':-^60}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")
    print(f"  Cost per product: ${avg_tokens * 0.00015 / 1000:.6f}")


def main():
    """Main execution function with systematic system"""
    print("=" * 80)
    print("SYSTEMATIC ENHANCED LLM-DRIVEN CLASSIFICATION SYSTEM")
    print("WITH PRODUCT TYPE ANALYSIS FRAMEWORK")
    print("=" * 80)

    print(f"Looking for master categories file: {MASTER_CATEGORIES_FILE}")
    print(f"Looking for YAML files in: {YAML_DIRECTORY}")

    try:
        # Test systematic classification
        print("\n1. Testing systematic classification...")
        test_success = test_systematic_classification()

        if test_success:
            print("\n" + "=" * 80)
            user_input = input(
                "Systematic tests passed! Proceed with validation sample processing? (y/n): "
            )

            if user_input.lower() == "y":
                validation_df = process_validation_sample()
                print("\n" + "=" * 80)
                print("🎉 SYSTEMATIC VALIDATION COMPLETE! 🎉")
                print("=" * 80)

                # Ask about full processing
                full_input = input("\nProceed with full dataset processing? (y/n): ")
                if full_input.lower() == "y":
                    process_full_dataset()
            else:
                print("Testing complete. You can run the validation later.")
        else:
            print("\n❌ Tests failed. Please check the error messages above.")

    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback

        traceback.print_exc()
        print(f"ERROR: {e}")


def process_full_dataset():
    """Process the full dataset with systematic classification"""
    logger.info("Starting full dataset processing with systematic classification...")

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df)} products from {INPUT_CSV}")

        # Add all classification columns including systematic analysis
        new_columns = [
            "primary_domain",
            "primary_subcategory",
            "primary_subsubcategory",
            "primary_subsubsubcategory",
            "primary_confidence",
            "primary_fit_score",
            "primary_path_valid",
            "domain_selection_confidence",
            "domain_selection_reasoning",
            "validated_path",
            "total_token_usage",
            "classification_count",
            "is_dual_function",
            "dual_function_pattern",
            "dual_function_reasoning",
            "secondary_domain",
            "secondary_subcategory",
            "secondary_subsubcategory",
            "secondary_subsubsubcategory",
            "secondary_confidence",
            "secondary_fit_score",
            "secondary_path_valid",
            "secondary_validated_path",
            "systematic_product_type",
            "systematic_confidence",
            "systematic_validation_applied",
        ]

        for col in new_columns:
            if col in [
                "primary_fit_score",
                "secondary_fit_score",
                "total_token_usage",
                "classification_count",
                "systematic_confidence",
            ]:
                df[col] = 0
            elif col in [
                "primary_path_valid",
                "secondary_path_valid",
                "is_dual_function",
            ]:
                df[col] = False
            else:
                df[col] = ""

        # Process in batches
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))

            logger.info(
                f"Processing batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx}) with systematic classification"
            )

            for idx in tqdm(
                range(start_idx, end_idx), desc=f"Systematic Batch {batch_num + 1}"
            ):
                name = df.at[idx, "Name"]
                description = (
                    df.at[idx, "Description"] if "Description" in df.columns else ""
                )

                try:
                    result = classifier.classify_product(name, description)

                    # Store primary results
                    primary = result.get("primary_classification", {})
                    df.at[idx, "primary_domain"] = primary.get("domain", "")
                    df.at[idx, "primary_subcategory"] = primary.get("subcategory", "")
                    df.at[idx, "primary_subsubcategory"] = primary.get(
                        "subsubcategory", ""
                    )
                    df.at[idx, "primary_subsubsubcategory"] = primary.get(
                        "subsubsubcategory", ""
                    )
                    df.at[idx, "primary_confidence"] = primary.get("confidence", "")
                    df.at[idx, "primary_path_valid"] = primary.get(
                        "is_valid_path", False
                    )
                    df.at[idx, "validated_path"] = primary.get("validated_path", "")
                    df.at[idx, "total_token_usage"] = result.get("total_token_usage", 0)
                    df.at[idx, "classification_count"] = result.get(
                        "classification_count", 1
                    )
                    df.at[idx, "is_dual_function"] = result.get(
                        "is_dual_function", False
                    )

                    # Store systematic analysis results
                    systematic_analysis = primary.get("systematic_type_analysis", {})
                    df.at[idx, "systematic_product_type"] = systematic_analysis.get(
                        "identified_type", ""
                    )
                    df.at[idx, "systematic_confidence"] = systematic_analysis.get(
                        "confidence", 0
                    )
                    df.at[idx, "systematic_validation_applied"] = str(
                        primary.get("systematic_validation_applied", [])
                    )

                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")
                    df.at[idx, "primary_domain"] = "Error"

            # Save progress after each batch
            df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Saved progress after systematic batch {batch_num + 1}")

        logger.info(
            f"Full systematic dataset processing complete. Results saved to {OUTPUT_CSV}"
        )

        # Generate final report
        generate_final_systematic_report(df)

    except Exception as e:
        logger.error(f"Error in systematic full dataset processing: {e}")
        raise


def generate_final_systematic_report(df: pd.DataFrame):
    """Generate final processing report for systematic system"""
    print("\n" + "=" * 80)
    print("FINAL SYSTEMATIC CLASSIFICATION REPORT")
    print("=" * 80)

    total_products = len(df)
    classified_products = len(
        df[
            df["primary_domain"].notna()
            & (df["primary_domain"] != "")
            & (df["primary_domain"] != "Other")
            & (df["primary_domain"] != "Error")
        ]
    )

    print(f"Total products processed: {total_products:,}")
    print(
        f"Successfully classified: {classified_products:,} ({classified_products/total_products*100:.1f}%)"
    )

    # Systematic product type analysis
    print(f"\n{'SYSTEMATIC PRODUCT TYPE DISTRIBUTION':-^60}")
    type_counts = df["systematic_product_type"].value_counts()
    for product_type, count in type_counts.head(10).items():
        if product_type:  # Skip empty values
            percentage = count / total_products * 100
            print(
                f"  {product_type.replace('_', ' ').title():<35} {count:>7,} ({percentage:>5.1f}%)"
            )

    # Systematic validation fixes applied
    validation_fixes = df[df["systematic_validation_applied"] != "[]"]
    print(f"\n{'SYSTEMATIC VALIDATION FIXES':-^60}")
    print(
        f"  Products with systematic fixes: {len(validation_fixes):,} ({len(validation_fixes)/total_products*100:.1f}%)"
    )

    # Enhanced depth analysis
    def calculate_depth(row):
        depth = 0
        levels = [
            "primary_subcategory",
            "primary_subsubcategory",
            "primary_subsubsubcategory",
        ]
        for level in levels:
            value = row.get(level, "")
            if value and str(value) != "nan" and str(value).strip():
                depth += 1
            else:
                break
        return depth

    df["classification_depth"] = df.apply(calculate_depth, axis=1)
    avg_depth = df["classification_depth"].mean()
    depth_3_plus = len(df[df["classification_depth"] >= 3])

    print(f"\n{'SYSTEMATIC DEPTH ANALYSIS':-^60}")
    print(f"  Average classification depth: {avg_depth:.2f}")
    print(
        f"  Deep classifications (3+ levels): {depth_3_plus:,} ({depth_3_plus/total_products*100:.1f}%)"
    )

    # Domain distribution
    print(f"\n{'DOMAIN DISTRIBUTION':-^60}")
    domain_counts = df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(15).items():
        print(f"  {domain:<35} {count:>7,} ({count/total_products*100:>5.1f}%)")

    # Quality metrics
    valid_paths = len(df[df["primary_path_valid"] == True])
    high_confidence = len(df[df["primary_confidence"] == "High"])

    print(f"\n{'QUALITY ANALYSIS':-^60}")
    print(
        f"  Valid classification paths: {valid_paths:,} ({valid_paths/total_products*100:.1f}%)"
    )
    print(
        f"  High confidence classifications: {high_confidence:,} ({high_confidence/total_products*100:.1f}%)"
    )

    # Systematic confidence analysis
    avg_systematic_confidence = df["systematic_confidence"].mean()
    print(f"  Average systematic confidence: {avg_systematic_confidence:.1f}%")

    # Token usage and cost analysis
    total_tokens = df["total_token_usage"].sum()
    avg_tokens = df["total_token_usage"].mean()
    print(f"\n{'SYSTEMATIC COST ANALYSIS':-^60}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost: ${total_tokens * 0.00015 / 1000:.2f}")
    print(f"  Cost per product: ${avg_tokens * 0.00015 / 1000:.6f}")


if __name__ == "__main__":
    main()
