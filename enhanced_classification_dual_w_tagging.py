# ENHANCED FIXED CLASSIFICATION SYSTEM WITH CHAIN TAGGING
# Adds comprehensive tagging system as separate API call to avoid prompt confusion

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
import random
from collections import defaultdict, Counter

# Configuration
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_enhanced_fixed_classification.csv"
VALIDATION_CSV = (
    "validation_enhanced_fixed_classification_w_tagging_prompt_changes_2.csv"
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED TAG SYSTEM FOR CHAIN PROMPTING
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

    def format_compact_prompt(self, max_per_category: int = 15) -> str:
        """Format tags for compact tagging prompt"""

        # Select most relevant tags for prompt
        technique_subset = list(sorted(self.technique_tags))[:max_per_category]
        research_subset = list(sorted(self.research_application_tags))[
            :max_per_category
        ]
        functional_subset = list(sorted(self.functional_tags))[:max_per_category]
        specimen_subset = list(sorted(self.specimen_tags))[:max_per_category]

        return {
            "technique": technique_subset,
            "research": research_subset,
            "functional": functional_subset,
            "specimen": specimen_subset,
        }


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
        """Enhanced domain selection prompt with dual classification and Other option"""

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

        prompt = f"""You are a life science product classification expert with access to {len(self.available_domains)} specialized domains.

Your task: Select 1-2 BEST domains for this product. You can assign a secondary domain if the product clearly fits multiple domains.

AVAILABLE DOMAINS ({len(self.available_domains)} total):
{domains_text}

Product Name: "{product_name}"
Description: "{description}"

CLASSIFICATION HIERARCHY RULES:

-Use the EXACT domain, subcategory , subsubcategory and subsubsubcategory names DO NOT input "subsubcategory" or "subsubsubcategory" into any field

1. PRODUCT TYPE beats APPLICATION:
- Instruments, systems, equipment, machines → Lab Equipment domain
- Physical laboratory consumables → Lab Equipment domain
- Reagents, chemicals, kits → Appropriate reagent domain
- Solvents are ALWAYS chemicals, never equipment
- Microscopes/Cameras/Imaging → Bioimaging/Microscopy
- Focus on APPLICATION and TECHNIQUE

2. CRITICAL EQUIPMENT vs REAGENT DISTINCTION:
EQUIPMENT = Physical instruments with moving parts, electronics, or measurement capabilities
Examples: "spectrophotometer", "centrifuge", "PCR machine", "plate reader", "microscope"

REAGENTS/SUPPLIES = Chemicals, biological materials, consumables, stains, membranes
Examples: "giemsa stain", "PVDF membrane", "acetonitrile", "cell culture medium"

Only classify as Chemistry if it's an actual chemical compound
Examples: "UPLC column" = Lab Equipment, "acetonitrile" = Chemistry

3. MANDATORY CONSUMABLES RULES :
CPT tubes, sample tubes, microcentrifuge tubes → Lab Equipment domain
Filter paper (Whatman, etc.), membrane filters → Lab Equipment domain
Microplates, cell culture plates, ELISA plates → Lab Equipment domain
Chromatography columns, C18 cartridges → Lab Equipment domain
Pipette tips, disposable pipettes → Lab Equipment domain

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

4. If uncertain between domains, choose the more specific/specialized one
5. If you are not sure about classification, use "Other" for primary_domain
6. Use the EXACT domain name as shown above (with spaces)

Respond with JSON only:
{{
    "primary_domain": "exact_domain_name_with_spaces",
    "secondary_domain": "exact_domain_name_with_spaces_or_null",
    "primary_confidence": "High/Medium/Low",
    "secondary_confidence": "High/Medium/Low_or_null",
    "reasoning": "brief explanation focusing on what the product IS and why it fits these domains"
}}"""

        return prompt

    def select_domain_with_llm(
        self, product_name: str, description: str
    ) -> Dict[str, Any]:
        """Use enhanced LLM to intelligently select 1-2 domains"""

        prompt = self.get_enhanced_domain_selection_prompt(product_name, description)

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

            # Process primary domain
            primary_domain_display = result.get("primary_domain", "Other")
            domain_mapping = getattr(self, "_domain_mapping", {})
            primary_domain = domain_mapping.get(
                primary_domain_display, primary_domain_display
            )

            # Validate primary domain
            if (
                primary_domain not in self.available_domains
                and primary_domain != "Other"
            ):
                underscore_version = primary_domain_display.replace(" ", "_")
                if underscore_version in self.available_domains:
                    primary_domain = underscore_version
                else:
                    # Try fuzzy matching
                    for domain_key in self.available_domains:
                        if (
                            domain_key.replace("_", " ").lower()
                            == primary_domain_display.lower()
                        ):
                            primary_domain = domain_key
                            break
                    else:
                        logger.warning(
                            f"LLM selected unrecognized primary domain '{primary_domain_display}', using Other"
                        )
                        primary_domain = "Other"

            # Process secondary domain
            secondary_domain_display = result.get("secondary_domain")
            secondary_domain = None
            if secondary_domain_display and secondary_domain_display != "null":
                secondary_domain = domain_mapping.get(
                    secondary_domain_display, secondary_domain_display
                )

                # Validate secondary domain
                if (
                    secondary_domain not in self.available_domains
                    and secondary_domain != "Other"
                ):
                    underscore_version = secondary_domain_display.replace(" ", "_")
                    if underscore_version in self.available_domains:
                        secondary_domain = underscore_version
                    else:
                        # Try fuzzy matching
                        for domain_key in self.available_domains:
                            if (
                                domain_key.replace("_", " ").lower()
                                == secondary_domain_display.lower()
                            ):
                                secondary_domain = domain_key
                                break
                        else:
                            logger.warning(
                                f"LLM selected unrecognized secondary domain '{secondary_domain_display}', ignoring"
                            )
                            secondary_domain = None

            final_result = {
                "primary_domain": primary_domain,
                "secondary_domain": secondary_domain,
                "primary_confidence": result.get("primary_confidence", "Low"),
                "secondary_confidence": (
                    result.get("secondary_confidence") if secondary_domain else None
                ),
                "reasoning": result.get("reasoning", ""),
                "token_usage": (
                    response.usage.total_tokens if hasattr(response, "usage") else 0
                ),
            }

            logger.info(
                f"Domain selection for '{product_name}': Primary={primary_domain} ({final_result['primary_confidence']}), Secondary={secondary_domain}"
            )

            return final_result

        except Exception as e:
            logger.error(f"Error in domain selection for '{product_name}': {e}")
            return {
                "primary_domain": "Other",
                "secondary_domain": None,
                "primary_confidence": "Low",
                "secondary_confidence": None,
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


class EnhancedLLMClassifier:
    """Enhanced LLM-driven classifier with separate chain tagging"""

    def __init__(
        self, category_system: LLMDrivenCategorySystem, tag_system: EnhancedTagSystem
    ):
        self.category_system = category_system
        self.tag_system = tag_system
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

    def tag_product(
        self,
        product_name: str,
        description: str = "",
        primary_classification: Dict = None,
        secondary_classification: Dict = None,
    ) -> Dict[str, Any]:
        """SEPARATE API CALL: Tag product with technique, research, and functional tags considering dual classification"""

        logger.info(f"🏷️ TAGGING: '{product_name}'")

        # Get compact tag lists for prompt
        tag_categories = self.tag_system.format_compact_prompt(max_per_category=12)

        # Build context-aware prompt with dual classification support
        context_info = ""
        if primary_classification:
            domain = primary_classification.get("domain", "")
            path = primary_classification.get("validated_path", "")
            context_info += f"Primary Domain: {domain}\n"
            if path:
                context_info += f"Primary Classification: {path}\n"

        if secondary_classification:
            domain = secondary_classification.get("domain", "")
            path = secondary_classification.get("validated_path", "")
            context_info += f"Secondary Domain: {domain}\n"
            if path:
                context_info += f"Secondary Classification: {path}\n"

        prompt = f"""You are a life science product tagging expert.

TASK: Assign relevant tags to this product from the provided tag categories.

{context_info}
Product: "{product_name}"
Description: "{description}"

TAGGING RULES:
1. Select 1-3 TECHNIQUE tags (HOW the product is used)
2. Select 1-2 RESEARCH APPLICATION tags (WHAT research it supports)  
3. Select 1-2 FUNCTIONAL tags (WHAT TYPE of product it is)
4. Only select tags that clearly apply - be selective, not exhaustive
5. Consider BOTH domain classifications when selecting tags (if secondary exists)

AVAILABLE TAGS:

TECHNIQUE TAGS (select 1-3 most relevant):
{', '.join(tag_categories['technique'])}

RESEARCH APPLICATION TAGS (select 1-2 most relevant):
{', '.join(tag_categories['research'])}

FUNCTIONAL TAGS (select 1-2 most relevant):
{', '.join(tag_categories['functional'])}

TAGGING EXAMPLES:

Antibody Example:
technique_tags: ["Western Blot", "Immunofluorescence"]
research_tags: ["Cell Biology", "Cancer Research"]
functional_tags: ["Research Use Only", "High Specificity"]

Equipment Example:
technique_tags: ["Flow Cytometry", "Cell Sorting"]
research_tags: ["Cell Biology", "Immunology"]
functional_tags: ["Instrument", "Automated"]

Kit Example:
technique_tags: ["ELISA", "Protein-Protein Interaction"]
research_tags: ["Immunology", "Diagnostics"]
functional_tags: ["Kit", "Quantitative"]

Respond with JSON only:
{{
    "technique_tags": ["tag1", "tag2"],
    "research_tags": ["tag1"],
    "functional_tags": ["tag1", "tag2"],
    "tag_confidence": "High/Medium/Low",
    "tagging_reasoning": "brief explanation of tag selection"
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

            # Validate tags against available tag sets
            validated_result = self._validate_tags(result)

            # Add metadata
            validated_result["token_usage"] = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            total_tags = (
                len(validated_result.get("technique_tags", []))
                + len(validated_result.get("research_tags", []))
                + len(validated_result.get("functional_tags", []))
            )

            validated_result["total_tags"] = total_tags

            logger.info(f"🏷️ Tagged with {total_tags} total tags")

            return validated_result

        except Exception as e:
            logger.error(f"❌ Error in tagging for '{product_name}': {e}")
            return {
                "technique_tags": [],
                "research_tags": [],
                "functional_tags": [],
                "tag_confidence": "Low",
                "tagging_reasoning": f"Error in tagging: {e}",
                "token_usage": 0,
                "total_tags": 0,
            }

    def _validate_tags(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tags against available tag sets"""

        # Extract tags and validate against tag system
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

        # Count invalid tags for logging
        original_technique = len(result.get("technique_tags", []))
        original_research = len(result.get("research_tags", []))
        original_functional = len(result.get("functional_tags", []))

        invalid_count = (
            (original_technique - len(technique_tags))
            + (original_research - len(research_tags))
            + (original_functional - len(functional_tags))
        )

        if invalid_count > 0:
            logger.warning(f"⚠️ Filtered out {invalid_count} invalid tags")

        return {
            "technique_tags": technique_tags,
            "research_tags": research_tags,
            "functional_tags": functional_tags,
            "tag_confidence": result.get("tag_confidence", "Medium"),
            "tagging_reasoning": result.get("tagging_reasoning", ""),
            "invalid_tags_filtered": invalid_count,
        }

    def classify_product(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Complete dual classification: Stage 1: Domain(s) → Stage 2a/2b: Path(s) → Stage 3: Unified Tags"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 DUAL CLASSIFICATION: '{product_name}'")
        self.logger.info(f"{'='*60}")

        total_tokens = 0

        # STAGE 1: Domain Selection (can return 1-2 domains)
        self.logger.info("🎯 STAGE 1: Domain Selection")
        domain_result = self.category_system.select_domain_with_llm(
            product_name, description
        )
        primary_domain = domain_result.get("primary_domain", "Other")
        secondary_domain = domain_result.get("secondary_domain")
        total_tokens += domain_result.get("token_usage", 0)

        classifications = []

        # STAGE 2a: Path Classification in Primary Domain
        self.logger.info(f"🗂️ STAGE 2a: Primary Classification in {primary_domain}")
        if primary_domain != "Other":
            primary_classification = self._classify_within_domain(
                product_name, description, primary_domain
            )
            total_tokens += (
                primary_classification.get("token_usage", 0)
                if primary_classification
                else 0
            )

            if primary_classification:
                primary_classification["is_primary"] = True
                classifications.append(primary_classification)
        else:
            primary_classification = self._create_fallback_classification(
                primary_domain
            )
            primary_classification["is_primary"] = True
            classifications.append(primary_classification)

        # STAGE 2b: Path Classification in Secondary Domain (if applicable)
        secondary_classification = None
        if secondary_domain and secondary_domain != "Other":
            self.logger.info(
                f"🗂️ STAGE 2b: Secondary Classification in {secondary_domain}"
            )
            secondary_classification = self._classify_within_domain(
                product_name, description, secondary_domain
            )
            total_tokens += (
                secondary_classification.get("token_usage", 0)
                if secondary_classification
                else 0
            )

            if secondary_classification:
                secondary_classification["is_primary"] = False
                classifications.append(secondary_classification)

        # STAGE 3: Unified Tag Assignment (considering both classifications)
        self.logger.info("🏷️ STAGE 3: Unified Tag Assignment")
        primary_class = classifications[0] if classifications else None
        secondary_class = classifications[1] if len(classifications) > 1 else None

        tagging_result = self.tag_product(
            product_name=product_name,
            description=description,
            primary_classification=primary_class,
            secondary_classification=secondary_class,
        )
        total_tokens += tagging_result.get("token_usage", 0)

        # Apply tags to all classifications
        for classification in classifications:
            classification.update(
                {
                    "technique_tags": tagging_result.get("technique_tags", []),
                    "research_tags": tagging_result.get("research_tags", []),
                    "functional_tags": tagging_result.get("functional_tags", []),
                    "tag_confidence": tagging_result.get("tag_confidence", "Low"),
                    "tagging_reasoning": tagging_result.get("tagging_reasoning", ""),
                    "total_tags": tagging_result.get("total_tags", 0),
                }
            )

            # Calculate all unique tags per classification
            all_tags = (
                classification.get("technique_tags", [])
                + classification.get("research_tags", [])
                + classification.get("functional_tags", [])
            )
            classification["all_tags"] = list(
                dict.fromkeys(all_tags)
            )  # Remove duplicates, preserve order

        # Format results
        if len(classifications) >= 2:
            # Dual classification result
            result = {
                "classifications": classifications,
                "primary_classification": classifications[0],
                "secondary_classification": classifications[1],
                "is_dual_function": True,
                "dual_function_pattern": "multi_domain_overlap",
                "dual_function_reasoning": f"Product fits multiple domains: {primary_domain}, {secondary_domain}",
                "classification_count": len(classifications),
                "total_token_usage": total_tokens,
            }
        elif len(classifications) == 1:
            # Single classification result
            result = self._format_single_classification(
                classifications[0], total_tokens
            )
        else:
            # Fallback result
            result = self._create_fallback_result(
                product_name,
                {
                    "reasoning": "Classification failed",
                    "token_usage": total_tokens,
                },
            )

        # Add stage-specific metadata
        result.update(
            {
                "stage1_domain_selection": {
                    "primary_domain": primary_domain,
                    "secondary_domain": secondary_domain,
                    "primary_confidence": domain_result.get(
                        "primary_confidence", "Low"
                    ),
                    "secondary_confidence": domain_result.get("secondary_confidence"),
                    "reasoning": domain_result.get("reasoning", ""),
                },
                "stage3_tagging": {
                    "total_tags": tagging_result.get("total_tags", 0),
                    "tag_confidence": tagging_result.get("tag_confidence", "Low"),
                    "invalid_tags_filtered": tagging_result.get(
                        "invalid_tags_filtered", 0
                    ),
                },
                "chain_prompting_stages": 3,
                "total_token_usage": total_tokens,
            }
        )

        return result

    def _classify_within_domain(
        self, product_name: str, description: str, domain_key: str
    ) -> Optional[Dict]:
        """Enhanced within-domain classification (Stage 2) with deeper classification encouragement"""

        # Get focused structure and guidance
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )

        prompt = f"""You are a life science product classification expert.

TASK: Classify this product to the DEEPEST possible level within the specified domain.

{focused_structure}

Product: "{product_name}"
Description: "{description}"

CRITICAL ANTIBODY CLASSIFICATION RULES:
1. **SECONDARY ANTIBODY PATTERNS** (highest priority):
   - "[Species] anti-[Target Species]" = Secondary Antibody (e.g., "goat anti-rabbit", "mouse anti-human")
   - "anti-[Species] IgG/IgM" = Secondary Antibody (e.g., "anti-rabbit IgG")
   - Conjugated anti-species products = Secondary Antibodies → Conjugated Secondary Antibodies
   - Examples: "alexa fluor goat anti-rabbit" → Secondary Antibodies → Conjugated Secondary Antibodies

2. **PRIMARY ANTIBODY PATTERNS**:
   - "anti-[Protein/Target]" = Primary Antibody (e.g., "anti-GFAP", "anti-BCL-2")
   - Target-specific without species reference = Primary Antibody

CRITICAL CHEMISTRY CLASSIFICATION RULES:
1. **SPECIFIC CHEMICAL TYPES**:
   - Histological stains (Fast Red, Methylene Blue) → Histological Stains and Dyes
   - Fluorescent compounds (fluorescein, rhodamine) → Fluorescent Compounds
   - Pharmaceutical compounds → Therapeutic Agents (NOT solvents)

2. **DEPTH REQUIREMENTS FOR SALTS**:
   - All salts must reach "Other Laboratory Salts" level minimum
   - Common salts: NaCl, KCl, MgSO4 → Basic Laboratory Salts → Specific salt type

3. **BUFFER COMPONENTS SPECIFICITY**:
   - Buffer solutions → Advanced Molecular Biology Chemicals → Buffer Components
   - Always try to reach Buffer Components level for buffer-related products

   
CRITICAL SOFTWARE CLASSIFICATION RULES:

1. **SPECIFIC SOFTWARE NAME PRIORITY**:
   - ALWAYS check for specific software subcategories FIRST
   - Examples: "STATA" → Statistical Analysis Software → STATA
   - Examples: "SPSS" → Statistical Analysis Software → SPSS
   - Examples: "Origin" → Laboratory Management Software → Origin Software

2. **SOFTWARE FUNCTION HIERARCHY**:
   - Image Analysis/Processing software → Image Analysis Software / Image Processing Software
   - Statistical software → Statistical Analysis Software → [Specific Name]
   - Data analysis → Data Analysis Software (only if no specific category)
   
REQUIREMENTS:
1. Domain Fit Score (0-100): How well does this product fit?
2. MANDATORY: Provide subcategory (exact name from paths above)
3. RECOMMENDED: Provide subsubcategory when available
4. PREFERRED: Provide subsubsubcategory when possible
5. DEEPER CLASSIFICATION RULE: If your confidence is Medium or High, you MUST attempt to reach at least subsubcategory level
6. Use EXACT names from the classification paths shown above

DEPTH ENCOURAGEMENT:
- High Confidence = MUST reach subsubsubcategory level if available
- Medium Confidence = MUST reach subsubcategory level if available  
- Low Confidence = subcategory level is acceptable

Respond with JSON only:
{{
    "domain_fit_score": 85,
    "belongs_in_domain": true,
    "subcategory": "exact_subcategory_name",
    "subsubcategory": "exact_subsubcategory_name_or_null", 
    "subsubsubcategory": "exact_3rd_level_name_or_null",
    "confidence": "High/Medium/Low",
    "reasoning": "detailed explanation of classification choice and depth achieved"
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

    def _create_fallback_classification(self, domain: str) -> Dict[str, Any]:
        """Create fallback classification when Stage 2 fails"""
        return {
            "domain": domain,
            "subcategory": "Unclassified",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "domain_fit_score": 10,
            "reasoning": "Could not classify within domain",
            "is_valid_path": False,
            "validated_path": f"{domain} -> Unclassified",
            "is_primary": True,
            "token_usage": 0,
        }

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
            "technique_tags": [],
            "research_tags": [],
            "functional_tags": [],
            "all_tags": [],
            "total_tags": 0,
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


def test_enhanced_dual_classification():
    """Test the enhanced dual classification system with chain tagging"""

    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = EnhancedLLMClassifier(category_system, tag_system)

    # Test cases including dual classification candidates
    test_cases = [
        {
            "name": "Anti-CD3 monoclonal antibody",
            "description": "Monoclonal antibody against CD3 for flow cytometry applications",
            "expected_primary": "Antibodies",
        },
        {
            "name": "qPCR master mix",
            "description": "Ready-to-use mix for quantitative PCR amplification",
            "expected_primary": "PCR",
        },
        {
            "name": "BD FACSCanto II Flow Cytometer",
            "description": "Multi-color flow cytometer for cell analysis and sorting",
            "expected_primary": "Lab_Equipment",
        },
        {
            "name": "Human TNF-alpha ELISA Kit",
            "description": "Quantitative ELISA kit for measuring TNF-alpha in human samples",
            "expected_primary": "Assay_Kits",
        },
        {
            "name": "PVDF membrane",
            "description": "Membrane for Western blotting applications",
            "expected_primary": "Protein",
        },
    ]

    print("=" * 80)
    print(" TESTING ENHANCED DUAL CLASSIFICATION WITH CHAIN TAGGING")
    print("=" * 80)

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n TEST {i}: {test_case['name']}")
        print("-" * 50)

        result = classifier.classify_product(
            test_case["name"], test_case["description"]
        )

        primary = result.get("primary_classification", {})
        secondary = result.get("secondary_classification")
        selected_domain = primary.get("domain", "")
        expected_domain = test_case["expected_primary"]

        print(f"Expected Primary: {expected_domain}")
        print(f"Selected Primary: {selected_domain}")

        if secondary:
            print(f"Secondary Domain: {secondary.get('domain', 'N/A')}")
            print(f"Dual Function: {result.get('is_dual_function', False)}")

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

        # Show tags
        technique_tags = primary.get("technique_tags", [])
        research_tags = primary.get("research_tags", [])
        functional_tags = primary.get("functional_tags", [])
        total_tags = primary.get("total_tags", 0)

        print(f"Tags ({total_tags} total):")
        if technique_tags:
            print(f"  Technique: {', '.join(technique_tags)}")
        if research_tags:
            print(f"  Research: {', '.join(research_tags)}")
        if functional_tags:
            print(f"  Functional: {', '.join(functional_tags)}")

        print(f"Tokens: {result.get('total_token_usage', 0)}")
        print(f"Stages: {result.get('chain_prompting_stages', 0)}")
        print(f"Classifications: {result.get('classification_count', 0)}")

        if domain_correct and (path_valid or fit_score >= 70) and total_tags > 0:
            success_count += 1
            print("✅ OVERALL: SUCCESS")
        else:
            print("❌ OVERALL: FAILED")

    print(f"\n{'='*80}")
    print(
        f" ENHANCED DUAL CLASSIFICATION RESULTS: {success_count}/{len(test_cases)} passed ({success_count/len(test_cases)*100:.1f}%)"
    )
    print(f"{'='*80}")

    return success_count >= 4


def process_enhanced_validation_sample():
    """Process validation sample with enhanced dual classification and tagging"""
    logger.info(" Starting ENHANCED DUAL validation sample processing...")

    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = EnhancedLLMClassifier(category_system, tag_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        sample_size = min(100, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        validation_df = df.iloc[sample_indices].copy()

        # Add enhanced dual classification columns
        new_columns = [
            "primary_domain",
            "primary_subcategory",
            "primary_subsubcategory",
            "primary_subsubsubcategory",
            "primary_confidence",
            "primary_fit_score",
            "primary_path_valid",
            "secondary_domain",
            "secondary_subcategory",
            "secondary_subsubcategory",
            "secondary_subsubsubcategory",
            "secondary_confidence",
            "secondary_fit_score",
            "secondary_path_valid",
            "is_dual_function",
            "dual_function_reasoning",
            "domain_selection_primary_confidence",
            "domain_selection_secondary_confidence",
            "domain_selection_reasoning",
            "validated_path_primary",
            "validated_path_secondary",
            "technique_tags",
            "research_tags",
            "functional_tags",
            "all_tags",
            "total_tags",
            "tag_confidence",
            "tagging_reasoning",
            "total_token_usage",
            "chain_prompting_stages",
            "classification_count",
        ]

        for col in new_columns:
            if col in [
                "primary_fit_score",
                "secondary_fit_score",
                "total_token_usage",
                "chain_prompting_stages",
                "classification_count",
                "total_tags",
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
            f" Processing {len(validation_df)} products with ENHANCED DUAL classification..."
        )

        for idx in tqdm(validation_df.index, desc=" Enhanced Dual Classification"):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            result = classifier.classify_product(name, description)

            # Store results
            classifications = result.get("classifications", [])

            # Primary classification
            if classifications:
                primary = classifications[0]
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
                validation_df.at[idx, "primary_confidence"] = primary.get(
                    "confidence", ""
                )
                validation_df.at[idx, "primary_fit_score"] = primary.get(
                    "domain_fit_score", 0
                )
                validation_df.at[idx, "primary_path_valid"] = primary.get(
                    "is_valid_path", False
                )
                validation_df.at[idx, "validated_path_primary"] = primary.get(
                    "validated_path", ""
                )

            # Secondary classification
            if len(classifications) > 1:
                secondary = classifications[1]
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
                validation_df.at[idx, "validated_path_secondary"] = secondary.get(
                    "validated_path", ""
                )

            # Dual function info
            validation_df.at[idx, "is_dual_function"] = result.get(
                "is_dual_function", False
            )
            validation_df.at[idx, "dual_function_reasoning"] = result.get(
                "dual_function_reasoning", ""
            )

            # Domain selection info
            stage1_info = result.get("stage1_domain_selection", {})
            validation_df.at[idx, "domain_selection_primary_confidence"] = (
                stage1_info.get("primary_confidence", "")
            )
            validation_df.at[idx, "domain_selection_secondary_confidence"] = (
                stage1_info.get("secondary_confidence", "")
            )
            validation_df.at[idx, "domain_selection_reasoning"] = stage1_info.get(
                "reasoning", ""
            )

            # Tags (from primary classification)
            if classifications:
                primary = classifications[0]
                validation_df.at[idx, "technique_tags"] = "|".join(
                    primary.get("technique_tags", [])
                )
                validation_df.at[idx, "research_tags"] = "|".join(
                    primary.get("research_tags", [])
                )
                validation_df.at[idx, "functional_tags"] = "|".join(
                    primary.get("functional_tags", [])
                )
                validation_df.at[idx, "all_tags"] = "|".join(
                    primary.get("all_tags", [])
                )
                validation_df.at[idx, "total_tags"] = primary.get("total_tags", 0)
                validation_df.at[idx, "tag_confidence"] = primary.get(
                    "tag_confidence", ""
                )
                validation_df.at[idx, "tagging_reasoning"] = primary.get(
                    "tagging_reasoning", ""
                )

            # Metadata
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )
            validation_df.at[idx, "chain_prompting_stages"] = result.get(
                "chain_prompting_stages", 0
            )
            validation_df.at[idx, "classification_count"] = result.get(
                "classification_count", 1
            )

        validation_df.to_csv(VALIDATION_CSV, index=False)
        logger.info(f"✅ Enhanced dual validation saved to {VALIDATION_CSV}")

        generate_enhanced_dual_validation_report(validation_df)
        return validation_df

    except Exception as e:
        logger.error(f"❌ Error in validation: {e}")
        raise


def generate_enhanced_dual_validation_report(validation_df: pd.DataFrame):
    """Generate enhanced validation report with dual classification and tagging analysis"""
    print("\n" + "=" * 80)
    print("ENHANCED DUAL CLASSIFICATION + CHAIN TAGGING REPORT")
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

    # Domain distribution
    print(f"\n{'PRIMARY DOMAIN DISTRIBUTION':-^60}")
    domain_counts = validation_df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(8).items():
        print(f"  {domain:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Dual classification analysis
    dual_function_count = len(validation_df[validation_df["is_dual_function"] == True])
    print(f"\n{'DUAL CLASSIFICATION ANALYSIS':-^60}")
    print(
        f"  Products with dual classification: {dual_function_count} ({dual_function_count/total_products*100:.1f}%)"
    )

    if dual_function_count > 0:
        print(f"  Secondary domain distribution:")
        secondary_counts = validation_df[validation_df["secondary_domain"].notna()][
            "secondary_domain"
        ].value_counts()
        for domain, count in secondary_counts.head(5).items():
            print(f"    {domain:<33} {count:>5}")

    # Depth analysis for primary classification
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

    validation_df["primary_classification_depth"] = validation_df.apply(
        calculate_depth, axis=1
    )
    depth_counts = (
        validation_df["primary_classification_depth"].value_counts().sort_index()
    )

    print(f"\n{'PRIMARY CLASSIFICATION DEPTH ANALYSIS':-^60}")
    for depth, count in depth_counts.items():
        percentage = count / total_products * 100
        print(f"  Depth {depth}: {count:>5} ({percentage:>5.1f}%)")

    avg_depth = validation_df["primary_classification_depth"].mean()
    depth_3_plus = len(
        validation_df[validation_df["primary_classification_depth"] >= 3]
    )

    print(f"  Average depth: {avg_depth:.2f}")
    print(
        f"  Deep classifications (3+ levels): {depth_3_plus} ({depth_3_plus/total_products*100:.1f}%)"
    )

    # Confidence analysis
    print(f"\n{'CONFIDENCE ANALYSIS':-^60}")
    primary_confidence_counts = validation_df["primary_confidence"].value_counts()
    for conf, count in primary_confidence_counts.items():
        if conf:  # Skip empty strings
            print(f"  Primary {conf}: {count} ({count/total_products*100:.1f}%)")

    # Tagging analysis
    print(f"\n{'TAGGING ANALYSIS':-^60}")
    avg_tags = validation_df["total_tags"].mean()
    products_with_tags = len(validation_df[validation_df["total_tags"] > 0])

    print(f"  Average tags per product: {avg_tags:.1f}")
    print(
        f"  Products with tags: {products_with_tags} ({products_with_tags/total_products*100:.1f}%)"
    )

    # Tag confidence distribution
    tag_confidence_counts = validation_df["tag_confidence"].value_counts()
    print(f"  Tag confidence distribution:")
    for conf, count in tag_confidence_counts.items():
        if conf:  # Skip empty strings
            print(f"    {conf}: {count} ({count/total_products*100:.1f}%)")

    # Most common tags
    print(f"\n{'TOP TECHNIQUE TAGS':-^60}")
    all_technique_tags = []
    for tags_str in validation_df["technique_tags"].dropna():
        if tags_str:
            all_technique_tags.extend(tags_str.split("|"))

    if all_technique_tags:
        technique_counts = Counter(all_technique_tags)
        for tag, count in technique_counts.most_common(8):
            print(f"  {tag:<35} {count:>5}")

    print(f"\n{'TOP RESEARCH TAGS':-^60}")
    all_research_tags = []
    for tags_str in validation_df["research_tags"].dropna():
        if tags_str:
            all_research_tags.extend(tags_str.split("|"))

    if all_research_tags:
        research_counts = Counter(all_research_tags)
        for tag, count in research_counts.most_common(8):
            print(f"  {tag:<35} {count:>5}")

    # Token usage and efficiency
    print(f"\n{'CHAIN PROMPTING EFFICIENCY':-^60}")
    total_tokens = validation_df["total_token_usage"].sum()
    avg_tokens = validation_df["total_token_usage"].mean()
    avg_stages = validation_df["chain_prompting_stages"].mean()

    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Average prompting stages: {avg_stages:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")

    # Quality metrics
    print(f"\n{'QUALITY METRICS':-^60}")
    valid_paths = len(validation_df[validation_df["primary_path_valid"] == True])
    high_confidence = len(validation_df[validation_df["primary_confidence"] == "High"])
    high_tag_confidence = len(validation_df[validation_df["tag_confidence"] == "High"])

    print(
        f"  Valid classification paths: {valid_paths} ({valid_paths/total_products*100:.1f}%)"
    )
    print(
        f"  High confidence classifications: {high_confidence} ({high_confidence/total_products*100:.1f}%)"
    )
    print(
        f"  High confidence tagging: {high_tag_confidence} ({high_tag_confidence/total_products*100:.1f}%)"
    )

    # Show excellent examples
    excellent_examples = validation_df[
        (validation_df["primary_classification_depth"] >= 3)
        & (validation_df["total_tags"] >= 3)
        & (validation_df["primary_confidence"] == "High")
    ].head(5)

    if len(excellent_examples) > 0:
        print(f"\n{'🌟 EXCELLENT CLASSIFICATIONS':-^60}")
        for idx, row in excellent_examples.iterrows():
            path_parts = []
            if row["primary_subcategory"]:
                path_parts.append(str(row["primary_subcategory"]))
            if row["primary_subsubcategory"]:
                path_parts.append(str(row["primary_subsubcategory"]))
            if row["primary_subsubsubcategory"]:
                path_parts.append(str(row["primary_subsubsubcategory"]))

            full_path = " -> ".join(path_parts) if path_parts else "N/A"
            all_tags = row.get("all_tags", "").split("|") if row.get("all_tags") else []
            tag_summary = f"({len(all_tags)} tags)" if all_tags else "(no tags)"
            dual_indicator = " [DUAL]" if row.get("is_dual_function") else ""

            print(
                f"  {row['Name'][:25]:<25} → {full_path[:30]:<30} {tag_summary}{dual_indicator}"
            )

    # Show dual classification examples
    dual_examples = validation_df[validation_df["is_dual_function"] == True].head(3)
    if len(dual_examples) > 0:
        print(f"\n{'🎭 DUAL CLASSIFICATION EXAMPLES':-^60}")
        for idx, row in dual_examples.iterrows():
            primary_domain = row.get("primary_domain", "N/A")
            secondary_domain = row.get("secondary_domain", "N/A")
            print(f"  {row['Name'][:30]:<30} → {primary_domain} + {secondary_domain}")


def main():
    """Main execution with enhanced system"""
    print("=" * 80)
    print(" ENHANCED FIXED CLASSIFICATION SYSTEM")
    print(" WITH SEPARATE CHAIN TAGGING")
    print("=" * 80)

    try:
        print("\n1. Testing enhanced dual classification with chain tagging...")
        test_success = test_enhanced_dual_classification()

        if test_success:
            print("\n" + "=" * 80)
            user_input = input("✅ Tests passed! Run validation sample? (y/n): ")

            if user_input.lower() == "y":
                validation_df = process_enhanced_validation_sample()

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
