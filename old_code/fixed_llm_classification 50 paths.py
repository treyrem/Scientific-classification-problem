# YAML-BASED LLM-DRIVEN CLASSIFICATION SYSTEM
# Removes rigid scoring, lets LLM choose domains intelligently

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
VALIDATION_CSV = "validation_llm_driven_classification.csv"
YAML_DIRECTORY = "C:/LabGit/150citations classification/category_structures"
MASTER_CATEGORIES_FILE = (
    "C:/LabGit/150citations classification/master_categories_new.yaml"
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


class LLMDrivenCategorySystem:
    """LLM-driven category system without rigid scoring"""

    def __init__(self, master_file: str = None, yaml_directory: str = None):
        self.master_file = master_file or MASTER_CATEGORIES_FILE
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.master_config = {}
        self.category_files = {}
        self.domain_structures = {}
        self.available_domains = []

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

    def get_domain_selection_prompt(self, product_name: str, description: str) -> str:
        """Create prompt for LLM to select the best domain"""

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

        prompt = f"""You are a life science product classification expert with access to 18 specialized domains.

Your task: Select the BEST domain for this product from the available domains below.

AVAILABLE DOMAINS ({len(self.available_domains)} total):
{domains_text}

Product Name: "{product_name}"
Description: "{description}"

CLASSIFICATION HIERARCHY RULES:
1. PRODUCT TYPE beats APPLICATION:
   - Instruments, systems, equipment, machines → Lab Equipment domain
   - Reagents, chemicals, kits → Appropriate reagent domain
   - Solvents are ALWAYS chemicals, never equipment
   
   
2. CONTEXT-SPECIFIC RULES:
   - Antibiotics in research context → Cell Biology or Cloning And Expression
   - Antibodies/markers → Antibodies domain
   - PCR/qPCR systems → Lab Equipment (not PCR domain)
   - Software/analysis tools → Software domain
   - Statistical software (SPSS, R, SAS) → Software domain
   - Organic solvents (acetonitrile, methanol) → Molecular Biology -> Chemicals -> Solvents
   - Fluorescent dyes (DyLight, Alexa Fluor) → Cell Biology -> Biomolecules -> Fluorophores, Dyes & Probes
   - ECM components (Matrigel, collagen) → Cell Biology -> Biomolecules -> Extracellular Matrix Components
   - Animal models (transgenic mice, knockout mice) → Cell Biology -> Animal Models
   - Apoptosis markers (cleaved caspase) → Antibodies -> Apoptosis Antibodies -> Caspase Antibodies
   - giemsa → Cell Biology (staining reagent, not equipment)
   - PVDF membrane → Protein (Western blot supply, not equipment)
   - immobilon → Protein (membrane brand, not equipment)


3. EXAMPLES:
   - "CFX384 PCR system" → "Lab Equipment" (it's equipment, not a PCR reagent)
   - "Streptomycin" → "Cell Biology" (research antibiotic, not therapeutic)
   - "Anti-CD3 antibody" → "Antibodies" (it's an antibody)
   - "PCR master mix" → "PCR" (it's a PCR reagent)
   - "ImageJ software" → "Software" (it's analysis software)
   ❌ NOT EQUIPMENT: membranes, stains, buffers,dyes

4. If uncertain between domains, choose the more specific/specialized one. Require strong evidence from description for domain assignment
5. Use "Other" if uncertain
6. Use the EXACT domain name as shown above (with spaces)

Respond with JSON only:
{{
    "selected_domain": "exact_domain_name_with_spaces",
    "confidence": "High/Medium/Low",
    "reasoning": "brief explanation focusing on what the product IS"
}}"""

        return prompt

    def select_domain_with_llm(
        self, product_name: str, description: str
    ) -> Dict[str, Any]:
        """Use LLM to intelligently select the best domain"""

        prompt = self.get_domain_selection_prompt(product_name, description)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
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

            logger.info(
                f"Domain selection for '{product_name}': {selected_domain} ({result.get('confidence', 'Unknown')})"
            )

            return result

        except Exception as e:
            logger.error(f"Error in domain selection for '{product_name}': {e}")
            return {
                "selected_domain": "Other",
                "confidence": "Low",
                "reasoning": f"Error in domain selection: {e}",
                "token_usage": 0,
            }

    def get_focused_structure_for_prompt(
        self, domain_key: str, product_name: str = "", description: str = ""
    ) -> str:
        """Enhanced structure prompt with complete deep paths and depth guidance"""

        if domain_key not in self.domain_structures:
            return f"Domain '{domain_key}' not found"

        # Get enhanced master category info
        domain_info = self.master_config.get("domain_mapping", {}).get(domain_key, {})

        lines = [f"DOMAIN: {domain_key.replace('_', ' ')}"]
        lines.append(f"Description: {domain_info.get('description', '')}")
        lines.append("")

        # Show COMPLETE deep paths from the YAML structure
        complete_paths = self._extract_actual_paths_from_yaml(domain_key)
        if complete_paths:
            lines.append(
                "COMPLETE CLASSIFICATION PATHS (use these exact names for deep classification):"
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

        # Enhanced depth requirements
        lines.append("DEPTH REQUIREMENTS:")
        lines.append("1. MANDATORY: Provide at least subcategory (level 1)")
        lines.append("2. STRONGLY RECOMMENDED: Provide subsubcategory (level 2)")
        lines.append("3. PREFERRED: Provide subsubsubcategory (level 3) when available")
        lines.append("4. Use EXACT names from the complete paths shown above")
        lines.append("5. Stop only when no deeper level exists in the paths")
        lines.append("")

        lines.append("CLASSIFICATION EXAMPLES:")
        lines.append("✅ GOOD: 'ATP Assay Kits' (2 levels)")
        lines.append("✅ BETTER: 'Cell-Based Assays -> ATP Assay Kits' (2 levels)")
        lines.append(
            "✅ BEST: 'Cell-Based Assays -> ATP Assay Kits -> Luciferase ATP Assays' (3 levels)"
        )
        lines.append("❌ SHALLOW: Only 'Cell-Based Assays' (1 level)")

        return "\n".join(lines)

    def _extract_actual_paths_from_yaml(self, domain_key: str) -> List[str]:
        """Extract complete deep classification paths from YAML structure"""
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

        # Return top 50 paths to show more deep options
        return unique_paths[:50]

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

                # Strategy 4: Keyword matching
                elif not found:
                    component_words = set(
                        component_lower.replace("-", " ").replace("_", " ").split()
                    )
                    best_match = None
                    best_score = 0

                    for key in current_node.keys():
                        key_words = set(
                            key.lower().replace("-", " ").replace("_", " ").split()
                        )
                        common_words = component_words.intersection(key_words)
                        if common_words:
                            score = len(common_words) / max(
                                len(component_words), len(key_words)
                            )
                            if (
                                score > best_score and score >= 0.3
                            ):  # At least 30% word overlap
                                best_score = score
                                best_match = key

                    if best_match:
                        validated_path.append(best_match)
                        current_node = current_node[best_match]
                        found = True
                        logger.info(
                            f"Keyword match: '{component}' -> '{best_match}' (score: {best_score:.2f})"
                        )

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
    """LLM-driven classifier without rigid scoring constraints"""

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
        """Enhanced classification with smart multi-domain detection"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CLASSIFYING: '{product_name}'")
        self.logger.info(f"{'='*60}")

        # Step 1: Get top domain candidates from LLM
        domain_candidates = self._get_multiple_domain_candidates(
            product_name, description
        )

        total_tokens = domain_candidates.get("token_usage", 0)

        # Step 2: Check if multi-domain classification is warranted
        top_domains = domain_candidates.get("ranked_domains", [])

        if len(top_domains) >= 2 and self._should_multi_classify(
            top_domains, product_name
        ):
            # MULTI-CLASSIFICATION: Product fits multiple domains
            return self._classify_multi_domain_product(
                product_name, description, top_domains, total_tokens
            )
        else:
            # SINGLE-CLASSIFICATION: Clear single domain winner
            selected_domain = top_domains[0]["domain"] if top_domains else "Other"
            return self._classify_single_domain_product(
                product_name, description, selected_domain, total_tokens
            )

    def _get_multiple_domain_candidates(
        self, product_name: str, description: str
    ) -> Dict[str, Any]:
        """Get ranked list of potential domains from LLM"""

        domain_info = []
        domain_mapping = {}

        for domain_key in self.available_domains:
            domain_data = self.master_config.get("domain_mapping", {}).get(
                domain_key, {}
            )
            domain_desc = domain_data.get("description", "")
            typical_products = domain_data.get("typical_products", [])
            keywords = domain_data.get("keywords", [])

            display_name = domain_key.replace("_", " ")
            domain_mapping[display_name] = domain_key

            key_keywords = ", ".join(keywords[:5]) if keywords else "N/A"

            domain_info.append(
                f"""
{display_name}:
  Description: {domain_desc}
  Key indicators: {key_keywords}
  Examples: {', '.join(typical_products[:2]) if typical_products else 'N/A'}"""
            )

        domains_text = "\n".join(domain_info)
        self._domain_mapping = domain_mapping

        prompt = f"""You are a life science product classification expert with access to 18 specialized domains.

Your task: Rank the TOP 3 most suitable domains for this product, including confidence scores.

AVAILABLE DOMAINS ({len(self.available_domains)} total):
{domains_text}

Product Name: "{product_name}"
Description: "{description}"

CLASSIFICATION HIERARCHY RULES:
1. PRODUCT TYPE beats APPLICATION:
   - Instruments, systems, equipment, machines → Lab Equipment domain
   - Reagents, chemicals, kits → Appropriate reagent domain
   
2. CONTEXT-SPECIFIC RULES:
   - Antibiotics in research context → Cell Biology or Cloning And Expression
   - Antibodies/markers → Antibodies domain
   - PCR/qPCR systems → Lab Equipment (not PCR domain)
   
3. Consider that some products legitimately fit multiple domains
4. Rank the top 3 domains that could reasonably classify this product
5. Give confidence scores (0-100) for each domain
6. Use "Other" only if no domains fit well

EXAMPLES:
- "CFX384 PCR system" could fit: Lab Equipment (90), PCR (70)
- "Streptomycin" could fit: Cell Biology (85), Cloning And Expression (75)
- "Anti-CD3 antibody" fits: Antibodies (95)

Respond with JSON only:
{{
    "ranked_domains": [
        {{"domain": "exact_domain_name_with_spaces", "confidence": 90, "reasoning": "why this domain fits"}},
        {{"domain": "second_domain_name", "confidence": 75, "reasoning": "secondary fit reason"}},
        {{"domain": "third_domain_name", "confidence": 60, "reasoning": "tertiary fit reason"}}
    ],
    "multi_domain_candidate": true/false,
    "primary_reasoning": "overall classification reasoning"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)
            result = json.loads(cleaned)

            # Convert domain names and validate
            ranked_domains = result.get("ranked_domains", [])
            validated_domains = []

            for domain_info in ranked_domains:
                display_name = domain_info.get("domain", "")
                domain_key = self._domain_mapping.get(
                    display_name, display_name.replace(" ", "_")
                )

                if domain_key in self.available_domains or domain_key == "Other":
                    validated_domains.append(
                        {
                            "domain": domain_key,
                            "confidence": domain_info.get("confidence", 0),
                            "reasoning": domain_info.get("reasoning", ""),
                        }
                    )

            result["ranked_domains"] = validated_domains
            result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            self.logger.info(
                f"Domain ranking for '{product_name}': {[d['domain'] for d in validated_domains]}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in domain ranking for '{product_name}': {e}")
            return {
                "ranked_domains": [
                    {"domain": "Other", "confidence": 10, "reasoning": f"Error: {e}"}
                ],
                "multi_domain_candidate": False,
                "primary_reasoning": f"Error in domain selection: {e}",
                "token_usage": 0,
            }

    def _should_multi_classify(
        self, top_domains: List[Dict], product_name: str
    ) -> bool:
        """Determine if product should be multi-classified"""
        if len(top_domains) < 2:
            return False

        primary_confidence = top_domains[0].get("confidence", 0)
        secondary_confidence = top_domains[1].get("confidence", 0)

        # Multi-classify if:
        # 1. Secondary domain has decent confidence (≥60)
        # 2. Gap between primary and secondary is small (≤30 points)
        # 3. Both domains are valid (not "Other")

        confidence_gap = primary_confidence - secondary_confidence

        should_multi = (
            secondary_confidence >= 75
            and confidence_gap <= 20
            and top_domains[0]["domain"] != "Other"
            and top_domains[1]["domain"] != "Other"
        )

        self.logger.info(
            f"Multi-classification check for '{product_name}': primary={primary_confidence}, secondary={secondary_confidence}, gap={confidence_gap}, decision={should_multi}"
        )

        return should_multi

    def _classify_multi_domain_product(
        self,
        product_name: str,
        description: str,
        top_domains: List[Dict],
        base_tokens: int,
    ) -> Dict[str, Any]:
        """Classify product in multiple domains"""

        self.logger.info(f"Multi-domain classification for: {product_name}")

        classifications = []
        total_tokens = base_tokens

        # Classify in top 2 domains
        for i, domain_info in enumerate(top_domains[:2]):
            domain_key = domain_info["domain"]
            is_primary = i == 0

            self.logger.info(
                f"Classifying in domain {i+1}: {domain_key} (primary={is_primary})"
            )

            classification = self._classify_within_domain(
                product_name, description, domain_key
            )

            if classification:
                classification.update(
                    {
                        "is_primary": is_primary,
                        "domain_confidence": domain_info.get("confidence", 0),
                        "domain_reasoning": domain_info.get("reasoning", ""),
                    }
                )
                classifications.append(classification)
                total_tokens += classification.get("token_usage", 0)

        # Format results
        if len(classifications) >= 2:
            return {
                "classifications": classifications,
                "primary_classification": classifications[0],
                "secondary_classification": classifications[1],
                "is_dual_function": True,
                "dual_function_pattern": "multi_domain_overlap",
                "dual_function_reasoning": f"Product fits multiple domains: {', '.join([d['domain'] for d in top_domains[:2]])}",
                "classification_count": len(classifications),
                "total_token_usage": total_tokens,
            }
        elif len(classifications) == 1:
            # Fall back to single classification
            return self._format_single_classification(classifications[0], total_tokens)
        else:
            # Fallback if all classifications failed
            return self._create_fallback_result(
                product_name,
                {
                    "reasoning": "Multi-domain classification failed",
                    "token_usage": base_tokens,
                },
            )

    def _classify_single_domain_product(
        self,
        product_name: str,
        description: str,
        selected_domain: str,
        base_tokens: int,
    ) -> Dict[str, Any]:
        """Classify product in single domain (existing logic)"""

        self.logger.info(
            f"Single-domain classification for: {product_name} in {selected_domain}"
        )

        if selected_domain == "Other":
            return self._create_fallback_result(
                product_name,
                {"reasoning": "No suitable domain found", "token_usage": base_tokens},
            )

        classification = self._classify_within_domain(
            product_name, description, selected_domain
        )
        total_tokens = base_tokens + (
            classification.get("token_usage", 0) if classification else 0
        )

        if classification:
            classification.update(
                {
                    "is_primary": True,
                    "domain_confidence": 95,  # High confidence for single domain
                    "domain_reasoning": f"Clear single domain match: {selected_domain}",
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

    def _classify_within_domain(
        self, product_name: str, description: str, domain_key: str
    ) -> Optional[Dict]:
        """Enhanced within-domain classification with deep path enforcement"""

        # Get focused structure and guidance
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        prompt = f"""You are a life science product classification expert.

    TASK: Classify this product to the DEEPEST possible level within the specified domain.

    {focused_structure}

    Product: "{product_name}"
    Description: "{description}"

    DEEP CLASSIFICATION REQUIREMENTS:
    1. Domain Fit Score (0-100): How well does this product fit in this domain?
    2. MANDATORY: Provide subcategory (exact name from paths above)
    3. STRONGLY RECOMMENDED: Provide subsubcategory (exact name from paths above)
    4. PREFERRED: Provide subsubsubcategory when a 3rd level exists
    5. Confidence Level: High/Medium/Low

    CRITICAL DEPTH RULES:
    - Use EXACT names from the complete classification paths shown above
    - Navigate as deep as the available paths allow
    - Only use null for subsubsubcategory if genuinely no 3rd level exists
    - Choose the most specific path that matches this product
    - Aim for 3 levels whenever possible

    DEPTH EXAMPLES:
    ❌ INSUFFICIENT: {{"subcategory": "Cell-Based Assays", "subsubcategory": null}}
    ✅ GOOD: {{"subcategory": "Cell-Based Assays", "subsubcategory": "ATP Assay Kits"}}
    ✅ EXCELLENT: {{"subcategory": "Cell-Based Assays", "subsubcategory": "ATP Assay Kits", "subsubsubcategory": "Luciferase ATP Assays"}}

    Respond with JSON only:
    {{
        "domain_fit_score": 75,
        "belongs_in_domain": true,
        "subcategory": "exact_subcategory_name",
        "subsubcategory": "exact_subsubcategory_name_or_null_if_none_exists", 
        "subsubsubcategory": "exact_3rd_level_name_or_null_if_none_exists",
        "confidence": "Medium",
        "reasoning": "detailed explanation of why this specific deep path was chosen",
        "classification_depth": 3
    }}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,  # Increased for deeper reasoning
            )

            text = response.choices[0].message.content.strip()
            cleaned = self.category_system._strip_code_fences(text)

            self.logger.info(f"LLM Response for {product_name}: {cleaned[:200]}...")

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

            # Log depth achievement
            self.logger.info(
                f"Classification depth achieved for '{product_name}': {depth_achieved} levels"
            )

            if depth_achieved < 2:
                self.logger.warning(
                    f"Shallow classification for '{product_name}': only {depth_achieved} levels"
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

            result.update(
                {
                    "domain": domain_key,
                    "is_valid_path": is_valid,
                    "validated_path": validated_path,
                    "is_primary": True,
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            self.logger.info(
                f"Domain analysis result: fit_score={result.get('domain_fit_score', 0)}, valid_path={is_valid}, depth={depth_achieved}"
            )

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error for '{product_name}': {text}")
            self.logger.error(f"Error details: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing domain {domain_key}: {e}")
            return None

    # ADD THIS NEW METHOD TO ENHANCE VALIDATION REPORTING
    def generate_validation_report(validation_df: pd.DataFrame):
        """Enhanced validation report with depth analysis"""
        print("\n" + "=" * 80)
        print("LLM-DRIVEN CLASSIFICATION VALIDATION REPORT")
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

        # Classification depth analysis
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

        validation_df["classification_depth"] = validation_df.apply(
            calculate_depth, axis=1
        )

        depth_counts = validation_df["classification_depth"].value_counts().sort_index()

        print(f"\n{'CLASSIFICATION DEPTH ANALYSIS':-^60}")
        for depth, count in depth_counts.items():
            percentage = count / total_products * 100
            print(f"  Depth {depth}: {count:>5} ({percentage:>5.1f}%)")

        avg_depth = validation_df["classification_depth"].mean()
        print(f"  Average depth: {avg_depth:.2f}")

        # Dual classification analysis
        dual_function_products = len(
            validation_df[validation_df["is_dual_function"] == True]
        )
        print(
            f"\nDual function products: {dual_function_products} ({dual_function_products/total_products*100:.1f}%)"
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

        # Confidence distribution
        print(f"\n{'CONFIDENCE DISTRIBUTION':-^60}")
        confidence_counts = validation_df["primary_confidence"].value_counts()
        for conf, count in confidence_counts.items():
            print(f"  {conf:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

        # Token usage
        total_tokens = validation_df["total_token_usage"].sum()
        avg_tokens = validation_df["total_token_usage"].mean()
        print(f"\n{'TOKEN USAGE ANALYSIS':-^60}")
        print(f"  Total tokens used: {total_tokens:,}")
        print(f"  Average tokens per product: {avg_tokens:.1f}")
        print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")

        # Deep classification examples
        deep_classifications = validation_df[
            validation_df["classification_depth"] >= 3
        ].head(5)
        if len(deep_classifications) > 0:
            print(f"\n{'EXCELLENT DEEP CLASSIFICATIONS (3+ levels)':-^60}")
            for idx, row in deep_classifications.iterrows():
                path_parts = []
                if row["primary_subcategory"]:
                    path_parts.append(str(row["primary_subcategory"]))
                if row["primary_subsubcategory"]:
                    path_parts.append(str(row["primary_subsubcategory"]))
                if row["primary_subsubsubcategory"]:
                    path_parts.append(str(row["primary_subsubsubcategory"]))

                full_path = " -> ".join(path_parts) if path_parts else "N/A"
                print(f"  {row['Name'][:35]:<35} → {full_path}")

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


def test_llm_driven_classification():
    """Test the LLM-driven classification system"""

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    # Test cases
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
    ]

    print("=" * 80)
    print("TESTING LLM-DRIVEN CLASSIFICATION SYSTEM")
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

        print(f"Expected Domain: {expected_domain}")
        print(f"Selected Domain: {selected_domain}")

        domain_correct = selected_domain == expected_domain
        path_valid = primary.get("is_valid_path", False)
        fit_score = primary.get("domain_fit_score", 0)

        print(f"Domain Selection: {'✓ CORRECT' if domain_correct else '✗ INCORRECT'}")
        print(f"Path Valid: {'✓ YES' if path_valid else '✗ NO'}")
        print(f"Fit Score: {fit_score}")
        print(f"Confidence: {primary.get('confidence', 'N/A')}")
        print(f"Validated Path: {primary.get('validated_path', 'N/A')}")
        print(f"Reasoning: {primary.get('reasoning', 'N/A')}")
        print(f"Token Usage: {result.get('total_token_usage', 0)}")

        if domain_correct and (path_valid or fit_score >= 70):
            success_count += 1
            print("✅ OVERALL: SUCCESS")
        else:
            print("❌ OVERALL: FAILED")

    print(f"\n{'='*80}")
    print(
        f"LLM-DRIVEN TEST RESULTS: {success_count}/{len(test_cases)} passed ({success_count/len(test_cases)*100:.1f}%)"
    )
    print(f"{'='*80}")

    return success_count >= 4


def process_validation_sample():
    """Process validation sample with LLM-driven classification"""
    logger.info("Starting LLM-driven validation sample processing...")

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        sample_size = min(100, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        validation_df = df.iloc[sample_indices].copy()

        # Add ALL classification columns including dual classification
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

        # Add missing dual classification columns
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

        logger.info(f"Processing {len(validation_df)} products...")

        for idx in tqdm(validation_df.index, desc="LLM-Driven Classification"):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            # Perform classification
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
            validation_df.at[idx, "primary_fit_score"] = primary.get(
                "domain_fit_score", 0
            )
            validation_df.at[idx, "primary_path_valid"] = primary.get(
                "is_valid_path", False
            )
            validation_df.at[idx, "domain_selection_confidence"] = primary.get(
                "domain_selection_confidence", ""
            )
            validation_df.at[idx, "domain_selection_reasoning"] = primary.get(
                "domain_selection_reasoning", ""
            )
            validation_df.at[idx, "validated_path"] = primary.get("validated_path", "")
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )

            # Store dual classification results
            validation_df.at[idx, "classification_count"] = result.get(
                "classification_count", 1
            )
            validation_df.at[idx, "is_dual_function"] = result.get(
                "is_dual_function", False
            )
            validation_df.at[idx, "dual_function_pattern"] = result.get(
                "dual_function_pattern", ""
            )
            validation_df.at[idx, "dual_function_reasoning"] = result.get(
                "dual_function_reasoning", ""
            )

            # Store secondary classification if it exists
            secondary = result.get("secondary_classification", {})
            if secondary:
                validation_df.at[idx, "secondary_domain"] = secondary.get("domain", "")
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
                validation_df.at[idx, "secondary_fit_score"] = secondary.get(
                    "domain_fit_score", 0
                )
                validation_df.at[idx, "secondary_path_valid"] = secondary.get(
                    "is_valid_path", False
                )
                validation_df.at[idx, "secondary_validated_path"] = secondary.get(
                    "validated_path", ""
                )

        # Save results
        validation_df.to_csv(VALIDATION_CSV, index=False)
        logger.info(f"LLM-driven validation sample saved to {VALIDATION_CSV}")

        # Generate report
        generate_validation_report(validation_df)

        return validation_df

    except Exception as e:
        logger.error(f"Error in validation processing: {e}")
        raise


def generate_validation_report(validation_df: pd.DataFrame):
    """Generate validation report"""
    print("\n" + "=" * 80)
    print("LLM-DRIVEN CLASSIFICATION VALIDATION REPORT")
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

    # Dual classification analysis
    dual_function_products = len(
        validation_df[validation_df["is_dual_function"] == True]
    )
    print(
        f"Dual function products: {dual_function_products} ({dual_function_products/total_products*100:.1f}%)"
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

    # Confidence distribution
    print(f"\n{'CONFIDENCE DISTRIBUTION':-^60}")
    confidence_counts = validation_df["primary_confidence"].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Token usage
    total_tokens = validation_df["total_token_usage"].sum()
    avg_tokens = validation_df["total_token_usage"].mean()
    print(f"\n{'TOKEN USAGE ANALYSIS':-^60}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("LLM-DRIVEN YAML-BASED CLASSIFICATION SYSTEM")
    print("=" * 80)

    print(f"Looking for master categories file: {MASTER_CATEGORIES_FILE}")
    print(f"Looking for YAML files in: {YAML_DIRECTORY}")

    try:
        # Test LLM-driven classification
        print("\n1. Testing LLM-driven classification...")
        test_success = test_llm_driven_classification()

        if test_success:
            print("\n" + "=" * 80)
            user_input = input(
                "Tests passed! Proceed with validation sample processing? (y/n): "
            )

            if user_input.lower() == "y":
                validation_df = process_validation_sample()
                print("\n" + "=" * 80)
                print("🎉 LLM-DRIVEN VALIDATION COMPLETE! 🎉")
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
    """Process the full dataset with LLM-driven classification"""
    logger.info("Starting full dataset processing with LLM-driven classification...")

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df)} products from {INPUT_CSV}")

        # Add all classification columns including dual classification
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
        ]

        for col in new_columns:
            if col in [
                "primary_fit_score",
                "secondary_fit_score",
                "total_token_usage",
                "classification_count",
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
                f"Processing batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx})"
            )

            for idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_num + 1}"):
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
                    df.at[idx, "primary_fit_score"] = primary.get("domain_fit_score", 0)
                    df.at[idx, "primary_path_valid"] = primary.get(
                        "is_valid_path", False
                    )
                    df.at[idx, "domain_selection_confidence"] = primary.get(
                        "domain_selection_confidence", ""
                    )
                    df.at[idx, "domain_selection_reasoning"] = primary.get(
                        "domain_selection_reasoning", ""
                    )
                    df.at[idx, "validated_path"] = primary.get("validated_path", "")
                    df.at[idx, "total_token_usage"] = result.get("total_token_usage", 0)
                    df.at[idx, "classification_count"] = result.get(
                        "classification_count", 1
                    )
                    df.at[idx, "is_dual_function"] = result.get(
                        "is_dual_function", False
                    )
                    df.at[idx, "dual_function_pattern"] = result.get(
                        "dual_function_pattern", ""
                    )
                    df.at[idx, "dual_function_reasoning"] = result.get(
                        "dual_function_reasoning", ""
                    )

                    # Store secondary classification if it exists
                    secondary = result.get("secondary_classification", {})
                    if secondary:
                        df.at[idx, "secondary_domain"] = secondary.get("domain", "")
                        df.at[idx, "secondary_subcategory"] = secondary.get(
                            "subcategory", ""
                        )
                        df.at[idx, "secondary_subsubcategory"] = secondary.get(
                            "subsubcategory", ""
                        )
                        df.at[idx, "secondary_subsubsubcategory"] = secondary.get(
                            "subsubsubcategory", ""
                        )
                        df.at[idx, "secondary_confidence"] = secondary.get(
                            "confidence", ""
                        )
                        df.at[idx, "secondary_fit_score"] = secondary.get(
                            "domain_fit_score", 0
                        )
                        df.at[idx, "secondary_path_valid"] = secondary.get(
                            "is_valid_path", False
                        )
                        df.at[idx, "secondary_validated_path"] = secondary.get(
                            "validated_path", ""
                        )

                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")
                    df.at[idx, "primary_domain"] = "Error"

            # Save progress after each batch
            df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Saved progress after batch {batch_num + 1}")

        logger.info(
            f"Full LLM-driven dataset processing complete. Results saved to {OUTPUT_CSV}"
        )

        # Generate final report
        generate_final_report(df)

    except Exception as e:
        logger.error(f"Error in full dataset processing: {e}")
        raise


def generate_final_report(df: pd.DataFrame):
    """Generate final processing report"""
    print("\n" + "=" * 80)
    print("FINAL LLM-DRIVEN CLASSIFICATION REPORT")
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

    # Dual classification analysis
    dual_function_products = len(df[df["is_dual_function"] == True])
    print(
        f"Dual function products: {dual_function_products:,} ({dual_function_products/total_products*100:.1f}%)"
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

    # Confidence distribution
    print(f"\n{'CONFIDENCE DISTRIBUTION':-^60}")
    confidence_counts = df["primary_confidence"].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf:<35} {count:>7,} ({count/total_products*100:>5.1f}%)")

    # Token usage and cost analysis
    total_tokens = df["total_token_usage"].sum()
    avg_tokens = df["total_token_usage"].mean()
    print(f"\n{'COST ANALYSIS':-^60}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost: ${total_tokens * 0.00015 / 1000:.2f}")

    # Top invalid path examples for debugging
    invalid_paths = df[df["primary_path_valid"] == False].head(5)
    if len(invalid_paths) > 0:
        print(f"\n{'INVALID PATH EXAMPLES (for debugging)':-^60}")
        for idx, row in invalid_paths.iterrows():
            print(f"  {row['Name'][:40]:<40} -> {row['validated_path']}")


if __name__ == "__main__":
    main()
