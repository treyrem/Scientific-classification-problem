# ENHANCED YAML-BASED LLM-DRIVEN CLASSIFICATION SYSTEM
# WITH FIXED VALIDATION LOGIC - MORE TRUSTING OF LLM SUGGESTIONS

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
VALIDATION_CSV = "validation_llm_driven_classification_fixed_3.csv"
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


class LLMDrivenCategorySystem:
    """Enhanced LLM-driven category system with FIXED validation logic"""

    def __init__(self, master_file: str = None, yaml_directory: str = None):
        self.master_file = master_file or MASTER_CATEGORIES_FILE
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.master_config = {}
        self.category_files = {}
        self.domain_structures = {}
        self.available_domains = []

        self.load_master_categories()
        self.load_individual_yaml_files()

        # Initialize domain name mapping AFTER loading structures
        self._initialize_domain_mapping()

        # Initialize trusted path patterns for LLM validation
        self._initialize_trusted_patterns()

    def _initialize_domain_mapping(self):
        """Initialize domain name mapping based on what's actually loaded"""
        self.domain_name_mapping = {}

        # Check for common domain name mismatches
        if (
            "Protein Biochemistry" in self.domain_structures
            and "Protein" not in self.domain_structures
        ):
            self.domain_name_mapping["Protein"] = "Protein Biochemistry"
            logger.info("🔧 Mapped 'Protein' -> 'Protein Biochemistry'")

        # Debug: Log what domains we actually have
        logger.info(f"🔍 Loaded domains: {list(self.domain_structures.keys())}")
        logger.info(f"🔧 Domain mappings: {self.domain_name_mapping}")

    def _initialize_trusted_patterns(self):
        """Initialize patterns we trust from LLM suggestions"""
        self.trusted_patterns = {
            "Lab_Equipment": {
                "first_components": [
                    "Analytical Instrumentation",
                    "Cell Analysis",
                    "Chromatography Equipment",
                    "Spectroscopy",
                ],
                "full_paths": [
                    ["Analytical Instrumentation", "Spectroscopy"],
                    ["Analytical Instrumentation", "Chromatography Equipment"],
                    ["Cell Analysis", "Flow Cytometry"],
                ],
            },
            "Antibodies": {
                "first_components": [
                    "Primary Antibodies",
                    "Secondary Antibodies",
                    "Research Area Antibodies",
                ],
                "full_paths": [
                    ["Primary Antibodies", "Monoclonal Antibodies"],
                    ["Primary Antibodies", "Application-Specific Primary Antibodies"],
                    ["Research Area Antibodies", "Neuroscience Antibodies"],
                ],
            },
            "Protein": {
                "first_components": [
                    "Western Blot Analysis",
                    "Biochemistry Reagents",
                    "Protein Purification",
                ],
                "full_paths": [
                    ["Western Blot Analysis", "Western Blot Supplies"],
                    ["Western Blot Analysis", "Western Blot Membranes"],
                    ["Biochemistry Reagents", "Bovine Serum Albumin"],
                ],
            },
            "Cell_Biology": {
                "first_components": [
                    "Biomolecules",
                    "Cell Analysis",
                    "Primary Cells, Cell Lines and Microorganisms",
                ],
                "full_paths": [
                    ["Biomolecules", "Proteins & Peptides"],
                    ["Biomolecules", "Fluorophores, Dyes & Probes"],
                ],
            },
            "Assay_Kits": {
                "first_components": [
                    "Cell-Based Assays",
                    "ELISA Kits",
                    "Quantitative Assays",
                    "Enzyme Assays",
                ],
                "full_paths": [
                    ["Cell-Based Assays", "Cell Viability Assay Kits"],
                    ["ELISA Kits", "Cytokine/Chemokine ELISA Kits"],
                ],
            },
        }

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

    def get_enhanced_domain_selection_prompt(
        self, product_name: str, description: str
    ) -> str:
        """Enhanced domain selection prompt with specific kit classification guidance"""

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

    -Use the EXACT domain, subcategory , subsubcategory and subsubsubcategory names DO NOT input "subsubcategory" or "subsubsubcategory" into any field
    
    1. PRODUCT TYPE beats APPLICATION:
    - Instruments, systems, equipment, machines → Lab Equipment domain
    - Reagents, chemicals, kits → Appropriate reagent domain
    - Solvents are ALWAYS chemicals, never equipment

    2. CRITICAL EQUIPMENT vs REAGENT DISTINCTION:
    EQUIPMENT = Physical instruments with moving parts, electronics, or measurement capabilities
    Examples: "spectrophotometer", "centrifuge", "PCR machine", "plate reader", "microscope"
    
    REAGENTS/SUPPLIES = Chemicals, biological materials, consumables, stains, membranes
    Examples: "giemsa stain", "PVDF membrane", "acetonitrile", "cell culture medium"

    3. **NEW: ENHANCED KIT CLASSIFICATION RULES**:
    
     **COMPLETE ASSAY KITS** → Assay_Kits domain:
     "ELISA kit", "cell viability assay kit", "caspase activity kit", "cytokine detection kit"
     "macsplex exosome kit", "ATP assay kit", "apoptosis detection kit"
    
      **PREPARATION/PURIFICATION KITS** → Specific technique domains:
     "DNA extraction kit" → Nucleic_Acid_Purification
     "plasmid maxi kit" → Cloning_And_Expression  
     "PCR master mix kit" → PCR
     "protein purification kit" → Protein
    
      **CELL PREPARATION KITS** → Cell_Biology:
     "fixation permeabilization kit" → Cell_Biology (NOT Immunochemicals)
     "cell isolation kit" → Cell_Biology (NOT Blood domain)
     "transfection kit" → Cell_Biology or Cloning_And_Expression
    CRITICAL CELL BIOLOGY CLASSIFICATION RULES:
    
     **CELL CULTURE MEDIA** (NOT stains):
    - "RPMI 1640", "DMEM", "medium", "media" → Cell Culture -> Cell Culture Media
    - "culture medium", "growth medium" → Cell Culture -> Cell Culture Media
    
     **CELL CULTURE CONSUMABLES** (NOT stains):
    - "plates", "dishes", "flasks", "carrier plates" → Cell Culture -> Cell Culture Consumables
    - "cell culture plates", "tissue culture" → Cell Culture -> Cell Culture Consumables
    
    **ACTUAL STAINS AND DYES**:
    - "giemsa", "crystal violet", "trypan blue" → Cell Culture -> Cell Culture Stains and Dyes
    - "fluorescent dye", "nuclear stain" → Cell Culture -> Cell Culture Stains and Dyes
    
    **DECISION TREE FOR CELL CULTURE PRODUCTS**:
    1. Is it liquid growth medium/media? → Cell Culture Media
    2. Is it plastic consumable (plates/dishes)? → Cell Culture Consumables  
    3. Is it actual staining reagent? → Cell Culture Stains and Dyes
    4. Is it cells themselves? → Primary Cells, Cell Lines and Microorganisms
    
    WRONG: "RPMI 1640 medium" → Cell Culture Stains and Dyes
    CORRECT: "RPMI 1640 medium" → Cell Culture Media
    
    WRONG: "CellCarrier plates" → Cell Culture Stains and Dyes  
    CORRECT: "CellCarrier plates" → Cell Culture Consumables (or Lab_Equipment)
    
    **NOT KITS** - Individual reagents:
    "RIPA buffer" → Protein (it's a buffer, NOT an assay kit)
    "lysis buffer" → Protein (buffer component, NOT complete kit)
       "antibody" → Antibodies (reagent, NOT kit)

    4. **KIT CLASSIFICATION DECISION TREE**:
    
    **Step 1**: Is it a complete, ready-to-use assay system?
    - YES → Assay_Kits domain
    - NO → Continue to Step 2
    
    **Step 2**: Is it for sample preparation/purification?
    - DNA/RNA → Nucleic_Acid_Purification
    - Protein → Protein  
    - Cloning → Cloning_And_Expression
    - Cells → Cell_Biology
    
    **Step 3**: Is it actually just a buffer/reagent with "assay" in the name?
    - "RIPA buffer" → Protein (extraction buffer)
    - "lysis buffer" → Protein (cell lysis reagent)
    - Individual antibodies → Antibodies

    5. CONTEXT-SPECIFIC RULES:
    - Antibiotics in research context → Cell Biology or Cloning And Expression
    - Antibodies/markers → Antibodies domain
    - PCR/qPCR systems → Lab Equipment (not PCR domain)
    - Software/analysis tools → Software domain

    6. COMMON MISCLASSIFICATION FIXES :
    **Staining Reagents** (NOT equipment):
    • "giemsa" → Cell Biology (cell staining reagent)
    • "crystal violet" → Cell Biology (bacterial staining)
    
    **Membranes & Supplies** (NOT equipment):
    • "PVDF membrane" → Protein (Western blot supply)
    • "immobilon" → Protein (membrane brand)
    
    **Solvents & Chemicals** (NOT equipment):
    • "acetonitrile" → Molecular Biology (HPLC solvent)
    • "methanol" → Molecular Biology (extraction solvent)

    7. **ENHANCED KIT EXAMPLES**:
      WRONG: "fixation permeabilization kit" → Immunochemicals
        Reasoning: "Contains immunoassay-related words"
      CORRECT: "fixation permeabilization kit" → Cell Biology
        Reasoning: "Kit for cell preparation/processing, not immunoassay detection"

      WRONG: "RIPA buffer" → Assay_Kits
        Reasoning: "Has 'assay' in the name"  
      CORRECT: "RIPA buffer" → Protein
        Reasoning: "RIPA is a protein extraction buffer, not a complete assay kit"

      CORRECT: "cell isolation kit" → Cell_Biology
        Reasoning: "Kit for cell preparation, even if targeting specific cell types"

      CORRECT: "ELISA kit" → Assay_Kits
        Reasoning: "Complete detection assay system"

    8. If uncertain between domains, choose the more specific/specialized one
    9. Use "Other" if uncertain
    10. Use the EXACT domain name as shown above (with spaces)

    Respond with JSON only:
    {{
        "selected_domain": "exact_domain_name_with_spaces",
        "confidence": "High/Medium/Low",
        "reasoning": "brief explanation focusing on what the product IS and why it fits this domain (kit type vs buffer vs complete system)"
    }}"""

        return prompt

    def select_domain_with_llm(
        self, product_name: str, description: str
    ) -> Dict[str, Any]:
        """Use enhanced LLM to intelligently select the best domain"""

        prompt = self.get_enhanced_domain_selection_prompt(product_name, description)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,  # Increased for comprehensive YAML context
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
        """Extract complete deep classification paths EXCLUDING structural terms"""
        if domain_key not in self.domain_structures:
            return []

        structure = self.domain_structures[domain_key]
        complete_paths = []

        # CRITICAL: Define structural terms to exclude
        STRUCTURAL_TERMS = {
            "subcategories",
            "subsubcategories",
            "subsubsubcategories",
            "categories",
            "category",
            "items",
            "list",
        }

        def extract_complete_paths(node, current_path=[], max_depth=4):
            if len(current_path) >= max_depth:
                return

            if isinstance(node, dict):
                for key, value in node.items():
                    # SKIP structural navigation terms
                    if key.lower() in STRUCTURAL_TERMS:
                        extract_complete_paths(value, current_path)
                        continue

                    new_path = current_path + [key]

                    # Only add paths with actual category names (not structural terms)
                    if len(new_path) >= 2 and key.lower() not in STRUCTURAL_TERMS:
                        display_path = " -> ".join(new_path[1:])
                        complete_paths.append(display_path)

                    if value and isinstance(value, (dict, list)):
                        extract_complete_paths(value, new_path)

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

    def _is_trusted_llm_path(self, domain_key: str, path_components: List[str]) -> bool:
        """Check if this LLM path matches our trusted patterns"""
        if not path_components or domain_key not in self.trusted_patterns:
            return False

        domain_patterns = self.trusted_patterns[domain_key]
        first_component = path_components[0]

        # Check if first component is trusted
        if first_component in domain_patterns["first_components"]:
            logger.info(
                f"🎯 First component '{first_component}' is trusted for {domain_key}"
            )
            return True

        # Check if full path matches a trusted pattern
        for trusted_path in domain_patterns["full_paths"]:
            if len(path_components) >= len(trusted_path):
                if path_components[: len(trusted_path)] == trusted_path:
                    logger.info(f" Full path matches trusted pattern: {trusted_path}")
                    return True

        return False

    def validate_classification_path(
        self, domain_key: str, path_components: List[str]
    ) -> Tuple[bool, str]:
        """FIXED validation that trusts good LLM suggestions"""

        # Handle domain name mapping
        actual_domain_key = self.domain_name_mapping.get(domain_key, domain_key)

        if actual_domain_key not in self.domain_structures:
            logger.error(
                f"❌ Domain '{actual_domain_key}' (mapped from '{domain_key}') not found"
            )
            logger.error(f"Available domains: {list(self.domain_structures.keys())}")
            return False, f"Domain '{domain_key}' not found"

        # Log mapping if it occurred
        if domain_key != actual_domain_key:
            logger.info(f" Domain mapped: {domain_key} -> {actual_domain_key}")

        structure = self.domain_structures[actual_domain_key]

        # Use original display name for output
        display_domain = domain_key.replace("_", " ")
        validated_path = [display_domain]

        logger.info(f" Validating path: {path_components} in domain {domain_key}")

        # If no path components, accept at domain level
        if not path_components or all(
            not comp or comp == "null" for comp in path_components
        ):
            final_path = " -> ".join(validated_path)
            logger.info(f"✅ Accepting domain-level classification: {final_path}")
            return True, final_path

        #  NEW: TRUST LLM FOR REASONABLE PATHS
        if len(path_components) >= 2 and self._is_trusted_llm_path(
            domain_key, path_components
        ):
            # Build full path trusting the LLM
            full_path = validated_path + path_components
            final_path = " -> ".join(full_path)
            logger.info(f" TRUSTING LLM suggestion: {final_path}")
            return True, final_path

        # Handle top-level domain structure navigation
        current_node = structure
        if isinstance(current_node, dict) and len(current_node) == 1:
            domain_name = list(current_node.keys())[0]
            current_node = current_node[domain_name]
            validated_path = [domain_name]  # Use actual domain name from YAML

        # Try normal validation for non-trusted paths
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
                component_lower = component.lower()

                # Strategy 1: Exact match
                if component in current_node:
                    validated_path.append(component)
                    current_node = current_node[component]
                    found = True
                    logger.info(f"✅ Exact match: '{component}'")

                # Strategy 2: Case-insensitive exact match
                elif not found:
                    for key in current_node.keys():
                        if key.lower() == component_lower:
                            validated_path.append(key)
                            current_node = current_node[key]
                            found = True
                            logger.info(
                                f"✅ Case-insensitive match: '{component}' -> '{key}'"
                            )
                            break

                # Strategy 3: Partial matching
                elif not found:
                    matches = []
                    for key in current_node.keys():
                        key_lower = key.lower()
                        if component_lower in key_lower or key_lower in component_lower:
                            score = max(len(component_lower), len(key_lower)) - abs(
                                len(component_lower) - len(key_lower)
                            )
                            matches.append((key, score))

                    if matches:
                        best_match = max(matches, key=lambda x: x[1])[0]
                        validated_path.append(best_match)
                        current_node = current_node[best_match]
                        found = True
                        logger.info(
                            f"✅ Partial match: '{component}' -> '{best_match}'"
                        )

                # Navigate deeper if we found a match
                if found and isinstance(current_node, dict):
                    if "subsubcategories" in current_node:
                        current_node = current_node["subsubcategories"]
                        logger.info(f"Navigated into subsubcategories level")

            elif isinstance(current_node, list):
                # Enhanced list matching
                component_lower = component.lower()

                # Try exact matches first
                for item in current_node:
                    if isinstance(item, str):
                        if item == component or item.lower() == component_lower:
                            validated_path.append(item)
                            found = True
                            logger.info(f"✅ List match: '{item}'")
                            break

                # Try partial matches if no exact match
                if not found:
                    for item in current_node:
                        if isinstance(item, str):
                            if (
                                component_lower in item.lower()
                                or item.lower() in component_lower
                            ):
                                validated_path.append(item)
                                found = True
                                logger.info(
                                    f" Partial list match: '{component}' -> '{item}'"
                                )
                                break

            if not found:
                # 🎯 IMPROVED: Be more lenient - if we got at least domain + one level, that's good
                if len(validated_path) >= 2:
                    logger.info(
                        f" Couldn't find '{component}' but have good partial path: {' -> '.join(validated_path)}"
                    )
                    break
                else:
                    logger.info(
                        f" No matches found for '{component}', accepting at domain level"
                    )
                    break

        # Always return True with best effort path
        final_path = " -> ".join(validated_path)
        logger.info(f"✅ Final validated path: {final_path}")
        return True, final_path

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


class LLMDrivenClassifier:
    """Enhanced LLM-driven classifier with improved validation"""

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
        self.logger.info(f"🚀 CLASSIFYING: '{product_name}'")
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

        prompt = f"""You are a life science product classification expert.

Your task: Rank the TOP 3 most suitable domains for this product.

AVAILABLE DOMAINS:
{domains_text}

Product Name: "{product_name}"
Description: "{description}"

RULES:
1. Instruments/equipment → Lab Equipment domain
2. Reagents/chemicals/kits → Appropriate reagent domain
3. Consider multi-domain products
4. Give confidence scores (0-100)

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
                max_tokens=500,
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
                f"🎯 Domain ranking: {[d['domain'] for d in validated_domains]}"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Error in domain ranking: {e}")
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

        confidence_gap = primary_confidence - secondary_confidence

        should_multi = (
            secondary_confidence >= 75
            and confidence_gap <= 20
            and top_domains[0]["domain"] != "Other"
            and top_domains[1]["domain"] != "Other"
        )

        self.logger.info(
            f"🤔 Multi-classification check: primary={primary_confidence}, secondary={secondary_confidence}, gap={confidence_gap}, decision={should_multi}"
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

        self.logger.info(f"🎭 Multi-domain classification: {product_name}")

        classifications = []
        total_tokens = base_tokens

        # Classify in top 2 domains
        for i, domain_info in enumerate(top_domains[:2]):
            domain_key = domain_info["domain"]
            is_primary = i == 0

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
            return self._format_single_classification(classifications[0], total_tokens)
        else:
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
        """Classify product in single domain"""

        self.logger.info(
            f"🎯 Single-domain classification: {product_name} in {selected_domain}"
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
                    "domain_confidence": 95,
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
        """Enhanced within-domain classification"""

        # Get focused structure and guidance
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        prompt = f"""You are a life science product classification expert.

TASK: Classify this product to the DEEPEST possible level within the specified domain.

{focused_structure}

Product: "{product_name}"
Description: "{description}"

REQUIREMENTS:
1. Domain Fit Score (0-100): How well does this product fit?
2. MANDATORY: Provide subcategory (exact name from paths above)
3. RECOMMENDED: Provide subsubcategory when available
4. PREFERRED: Provide subsubsubcategory when possible
5. Use EXACT names from the classification paths shown above

Respond with JSON only:
{{
    "domain_fit_score": 85,
    "belongs_in_domain": true,
    "subcategory": "exact_subcategory_name",
    "subsubcategory": "exact_subsubcategory_name_or_null", 
    "subsubsubcategory": "exact_3rd_level_name_or_null",
    "confidence": "High/Medium/Low",
    "reasoning": "detailed explanation of classification choice"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self.category_system._strip_code_fences(text)

            result = json.loads(cleaned)

            # Clean null values
            result = self._clean_classification_result(result)

            # Validate meaningful classification
            if not self._has_meaningful_classification(result):
                self.logger.warning(
                    f"⚠️ No meaningful classification for {product_name}"
                )
                return None

            # Calculate depth
            depth_achieved = self._calculate_classification_depth(result)
            result["depth_achieved"] = depth_achieved

            self.logger.info(
                f" Classification depth: {depth_achieved} levels for '{product_name}'"
            )

            # Validate the classification path
            path_components = self._extract_path_components(result)
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

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON parse error for '{product_name}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error analyzing domain {domain_key}: {e}")
            return None

    def _clean_classification_result(self, result: Dict) -> Dict:
        """Clean null values and structural terms"""

        NULL_VALUES = {None, "null", "", "None", "nan", "NULL"}
        STRUCTURAL_TERMS = {
            "subcategories",
            "subsubcategories",
            "subsubsubcategories",
            "categories",
        }

        for level in ["subcategory", "subsubcategory", "subsubsubcategory"]:
            value = result.get(level)
            str_value = str(value).strip() if value is not None else ""

            if (
                value in NULL_VALUES
                or str_value in NULL_VALUES
                or str_value.lower() in NULL_VALUES
                or str_value.lower() in STRUCTURAL_TERMS
            ):
                result[level] = ""
            else:
                result[level] = str_value

        return result

    def _has_meaningful_classification(self, result: Dict) -> bool:
        """Check if classification has meaningful content"""
        subcategory = result.get("subcategory", "").strip()
        return subcategory and subcategory not in {"", "null", "None", "subcategories"}

    def _calculate_classification_depth(self, result: Dict) -> int:
        """Calculate classification depth"""
        depth = 0
        for level in ["subcategory", "subsubcategory", "subsubsubcategory"]:
            value = result.get(level, "").strip()
            if value and value not in {"", "null", "None"}:
                depth += 1
            else:
                break
        return depth

    def _extract_path_components(self, result: Dict) -> List[str]:
        """Extract valid path components"""
        components = []
        for level in ["subcategory", "subsubcategory", "subsubsubcategory"]:
            value = result.get(level, "").strip()
            if value and value not in {"", "null", "None"}:
                components.append(value)
        return components

    def _create_fallback_result(
        self, product_name: str, domain_selection: Dict
    ) -> Dict[str, Any]:
        """Create fallback result when classification fails"""

        self.logger.warning(f"⚠️ Using fallback for: {product_name}")

        fallback_classification = {
            "domain": "Other",
            "subcategory": "Unclassified",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "domain_fit_score": 10,
            "reasoning": domain_selection.get("reasoning", "Could not classify"),
            "is_valid_path": False,
            "validated_path": "Other -> Unclassified",
            "is_primary": True,
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
    """Test the enhanced LLM-driven classification system"""

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
            "name": "PVDF membrane",
            "description": "Membrane for Western blotting",
            "expected_domain": "Protein",
        },
    ]

    print("=" * 80)
    print(" TESTING FIXED LLM-DRIVEN CLASSIFICATION SYSTEM")
    print("=" * 80)

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n TEST {i}: {test_case['name']}")
        print("-" * 50)

        result = classifier.classify_product(
            test_case["name"], test_case["description"]
        )

        primary = result.get("primary_classification", {})
        selected_domain = primary.get("domain", "")
        expected_domain = test_case["expected_domain"]

        print(f"Expected: {expected_domain}")
        print(f"Selected: {selected_domain}")

        domain_correct = selected_domain == expected_domain
        path_valid = primary.get("is_valid_path", False)
        fit_score = primary.get("domain_fit_score", 0)
        depth_achieved = primary.get("depth_achieved", 0)

        print(f"Domain: {'✅ CORRECT' if domain_correct else '❌ INCORRECT'}")
        print(f"Valid: {'✅ YES' if path_valid else '❌ NO'}")
        print(f"Fit Score: {fit_score}")
        print(f"Depth: {depth_achieved} levels")
        print(f"Confidence: {primary.get('confidence', 'N/A')}")
        print(f"Path: {primary.get('validated_path', 'N/A')}")
        print(f"Tokens: {result.get('total_token_usage', 0)}")

        if domain_correct and (path_valid or fit_score >= 70):
            success_count += 1
            print("✅ OVERALL: SUCCESS")
        else:
            print("❌ OVERALL: FAILED")

    print(f"\n{'='*80}")
    print(
        f" FIXED CLASSIFICATION RESULTS: {success_count}/{len(test_cases)} passed ({success_count/len(test_cases)*100:.1f}%)"
    )
    print(f"{'='*80}")

    return success_count >= 4


def process_validation_sample():
    """Process validation sample with fixed classification"""
    logger.info(" Starting FIXED validation sample processing...")

    category_system = LLMDrivenCategorySystem()
    classifier = LLMDrivenClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        sample_size = min(100, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        validation_df = df.iloc[sample_indices].copy()

        # Add classification columns
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
            "secondary_domain",
            "secondary_subcategory",
            "secondary_subsubcategory",
            "secondary_subsubsubcategory",
            "secondary_confidence",
            "secondary_fit_score",
            "secondary_path_valid",
            "secondary_validated_path",
            "is_dual_function",
            "dual_function_pattern",
            "dual_function_reasoning",
            "classification_count",
        ]

        for col in new_columns:
            if col in [
                "primary_fit_score",
                "secondary_fit_score",
                "total_token_usage",
                "classification_count",
            ]:
                validation_df[col] = 0
            elif col in [
                "primary_path_valid",
                "secondary_path_valid",
                "is_dual_function",
            ]:
                validation_df[col] = False
            else:
                validation_df[col] = ""

        logger.info(
            f" Processing {len(validation_df)} products with FIXED validation logic..."
        )

        for idx in tqdm(validation_df.index, desc=" Fixed Classification"):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            result = classifier.classify_product(name, description)

            # Store results
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
            validation_df.at[idx, "validated_path"] = primary.get("validated_path", "")
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )

            validation_df.at[idx, "classification_count"] = result.get(
                "classification_count", 1
            )
            validation_df.at[idx, "is_dual_function"] = result.get(
                "is_dual_function", False
            )

            # Store secondary classification
            secondary = result.get("secondary_classification", {})
            if secondary:
                validation_df.at[idx, "secondary_domain"] = secondary.get("domain", "")
                validation_df.at[idx, "secondary_subcategory"] = secondary.get(
                    "subcategory", ""
                )
                validation_df.at[idx, "secondary_subsubcategory"] = secondary.get(
                    "subsubcategory", ""
                )

        validation_df.to_csv(VALIDATION_CSV, index=False)
        logger.info(f"✅ Fixed validation saved to {VALIDATION_CSV}")

        generate_validation_report(validation_df)
        return validation_df

    except Exception as e:
        logger.error(f"❌ Error in validation: {e}")
        raise


def generate_validation_report(validation_df: pd.DataFrame):
    """Generate validation report"""
    print("\n" + "=" * 80)
    print("FIXED LLM-DRIVEN CLASSIFICATION VALIDATION REPORT")
    print("=" * 80)

    total_products = len(validation_df)
    classified_products = len(
        validation_df[
            validation_df["primary_domain"].notna()
            & (validation_df["primary_domain"] != "")
            & (validation_df["primary_domain"] != "Other")
        ]
    )

    print(f"Total products: {total_products}")
    print(
        f"Successfully classified: {classified_products} ({classified_products/total_products*100:.1f}%)"
    )

    # Depth analysis
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

    print(f"\n{'DEPTH ANALYSIS (FIXED)':-^60}")
    for depth, count in depth_counts.items():
        percentage = count / total_products * 100
        print(f"  Depth {depth}: {count:>5} ({percentage:>5.1f}%)")

    avg_depth = validation_df["classification_depth"].mean()
    depth_3_plus = len(validation_df[validation_df["classification_depth"] >= 3])

    print(f"  Average depth: {avg_depth:.2f}")
    print(
        f"  Deep classifications (3+ levels): {depth_3_plus} ({depth_3_plus/total_products*100:.1f}%)"
    )

    # Show excellent examples
    deep_classifications = validation_df[
        validation_df["classification_depth"] >= 3
    ].head(5)
    if len(deep_classifications) > 0:
        print(f"\n{'🌟 EXCELLENT DEEP CLASSIFICATIONS':-^60}")
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


def main():
    """Main execution with fixed system"""
    print("=" * 80)
    print(" FIXED LLM-DRIVEN CLASSIFICATION SYSTEM")
    print(" MORE TRUSTING VALIDATION LOGIC")
    print("=" * 80)

    try:
        print("\n1. Testing fixed classification...")
        test_success = test_llm_driven_classification()

        if test_success:
            print("\n" + "=" * 80)
            user_input = input("✅ Tests passed! Run validation sample? (y/n): ")

            if user_input.lower() == "y":
                validation_df = process_validation_sample()

                full_input = input("\n Proceed with full dataset? (y/n): ")
                if full_input.lower() == "y":
                    print(" Starting full dataset processing...")
                    # Add full dataset processing here
            else:
                print("Testing complete.")
        else:
            print("\n❌ Tests failed. Check the error messages above.")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
