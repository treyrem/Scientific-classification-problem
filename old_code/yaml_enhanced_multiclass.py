# YAML-BASED ENHANCED MULTI-CLASSIFICATION SYSTEM
# Enhances the existing YAML system with intelligent dual-function detection

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

# Configuration
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_yaml_enhanced_multiclass.csv"
VALIDATION_CSV = "validation_yaml_enhanced_multiclass.csv"
YAML_DIRECTORY = "C:/LabGit/150citations classification/category_structures"
MASTER_CATEGORIES_FILE = "C:/LabGit/150citations classification/master_categories.yaml"

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


# ENHANCED TAG SYSTEM - Restored from original system
class EnhancedTagSystem:
    """Enhanced tag system for life science products"""

    def __init__(self):
        self.technique_tags = {
            "Western Blot",
            "Immunohistochemistry",
            "Immunofluorescence",
            "Flow Cytometry",
            "ELISA",
            "Immunoprecipitation",
            "ChIP",
            "ELISPOT",
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
            "Confocal Microscopy",
            "Fluorescence Microscopy",
            "Phase Contrast",
            "DIC",
            "Super-Resolution Microscopy",
            "Electron Microscopy",
            "Light Sheet Microscopy",
            "TIRF Microscopy",
            "Multiphoton Microscopy",
            "Liquid Handling",
            "Automated Workstation",
            "High-Throughput Screening",
            "Robotic System",
            "Plate Reader",
            "Automated Imaging",
        }

        self.research_application_tags = {
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
            "Oncology",
            "Neurodegenerative Disease",
            "Cardiovascular Disease",
            "Autoimmune Disease",
            "Infectious Disease",
            "Metabolic Disease",
            "Genetic Disorders",
            "Rare Disease",
            "Drug Discovery",
            "Biomarker Discovery",
            "Diagnostics",
            "Companion Diagnostics",
            "Personalized Medicine",
            "Immunotherapy",
            "Gene Therapy",
            "Cell Therapy",
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

        self.functional_tags = {
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

    def validate_tags(self, tags: List[str], tag_type: str = "all") -> List[str]:
        """Validate tags against the appropriate tag set"""
        if tag_type == "technique":
            valid_tags = self.technique_tags
        elif tag_type == "research":
            valid_tags = self.research_application_tags
        elif tag_type == "functional":
            valid_tags = self.functional_tags
        elif tag_type == "specimen":
            valid_tags = self.specimen_tags
        else:
            valid_tags = self.all_tags

        return [tag for tag in tags if tag in valid_tags]

    def get_tag_lists_for_prompt(self) -> Dict[str, str]:
        """Get formatted tag lists for LLM prompts"""
        return {
            "technique_tags": ", ".join(list(sorted(self.technique_tags))[:25]),
            "research_tags": ", ".join(
                list(sorted(self.research_application_tags))[:20]
            ),
            "functional_tags": ", ".join(list(sorted(self.functional_tags))[:25]),
            "specimen_tags": ", ".join(list(sorted(self.specimen_tags))[:15]),
        }


# DUAL FUNCTION PATTERNS - Aligned with actual YAML structures
DUAL_FUNCTION_PATTERNS = {
    "cancer_apoptosis_markers": {
        "indicators": [
            r"\bbcl-?2\b",
            r"\bbcl2\b",
            r"\bp53\b",
            r"\btp53\b",
            r"\bbax\b",
            r"\bbak\b",
            r"\bc-myc\b",
            r"\bmyc\b",
            r"\bbrca1\b",
            r"\bbrca2\b",
            r"\bpten\b",
            r"\bmdm2\b",
            r"\bbid\b",
            r"\bbim\b",
            r"\bnoxa\b",
            r"\bpuma\b",
            r"\bbclxl\b",
            r"\bmcl1\b",
        ],
        "functions": ["cancer marker", "apoptosis regulator"],
        "primary_domain": "Antibodies",
        "classification_contexts": [
            "cancer biomarker and diagnostic marker",
            "apoptosis regulator and cell death pathway component",
        ],
        "reasoning": "Protein functions in both cancer progression and apoptosis regulation",
    },
    "nuclear_dna_stains": {
        "indicators": [
            r"\bdapi\b",
            r"\bhoechst\b",
            r"\bpropidium\s+iodide\b",
            r"\bpi\b",
            r"\b7-?aad\b",
            r"\bto-?pro\b",
            r"\bsytox\b",
        ],
        "functions": ["nuclear stain", "dna binding dye"],
        "primary_domain": "Cell_Biology",  # Has Biomolecules -> Fluorophores, Dyes & Probes
        "classification_contexts": [
            "nuclear morphology and cell identification stain",
            "DNA content analysis and flow cytometry reagent",
        ],
        "reasoning": "Used for both nuclear imaging and DNA quantification applications",
    },
    "protease_dissociation": {
        "indicators": [
            r"\btrypsin\b",
            r"\bcollagenase\b",
            r"\bdispase\b",
            r"\baccutase\b",
            r"\belastase\b",
            r"\bhyaluronidase\b",
        ],
        "functions": ["protease enzyme", "cell dissociation reagent"],
        "primary_domain": "Protein_Biochemistry",  # Has Biomolecules -> Enzymes
        "classification_contexts": [
            "protease enzyme for protein cleavage and analysis",
            "cell dissociation reagent for cell culture applications",
        ],
        "reasoning": "Functions as both specific protease and cell culture dissociation agent",
    },
    # REMOVED: solvent_cryoprotectant - No clear dual classification path in YAMLs
    # REMOVED: cytokine_growth_factors - Would need Proteins_Peptides domain which isn't loaded
    # REMOVED: fluorescent_reporter - Would need specific reporter protein categories
}


class EnhancedDualFunctionCategorySystem:
    """Enhanced category system with dual-function detection capabilities"""

    def __init__(self, master_file: str = None, yaml_directory: str = None):
        self.master_file = master_file or MASTER_CATEGORIES_FILE
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.master_config = {}
        self.category_files = {}
        self.domain_keywords = defaultdict(set)
        self.domain_subcategories = defaultdict(set)
        self.domain_structures = {}
        self.dual_function_index = {}

        self.load_master_categories()
        self.load_individual_yaml_files()
        self.build_comprehensive_keyword_index()
        self.build_dual_function_index()

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
        """Load individual YAML files and extract their actual content"""
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
                    logger.info(f"✓ Loaded {domain_key} from {yaml_file}")
                else:
                    logger.warning(f"⚠ File not found: {yaml_file}")

            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")

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

    def build_comprehensive_keyword_index(self):
        """Build comprehensive keyword index from actual YAML content"""
        for domain_key, structure in self.domain_structures.items():
            keywords = set()
            subcategories = set()

            # Add keywords from master config
            domain_info = self.master_config.get("domain_mapping", {}).get(
                domain_key, {}
            )
            master_keywords = domain_info.get("keywords", [])
            for keyword in master_keywords:
                keywords.update(self._tokenize_text(keyword))

            # Extract keywords from actual structure
            self._extract_keywords_from_structure(structure, keywords, subcategories)

            self.domain_keywords[domain_key] = keywords
            self.domain_subcategories[domain_key] = subcategories

            logger.info(
                f"Built index for {domain_key}: {len(keywords)} keywords, {len(subcategories)} subcategories"
            )

    def build_dual_function_index(self):
        """Build index of dual-function patterns mapped to YAML subcategories"""
        for pattern_name, pattern_info in DUAL_FUNCTION_PATTERNS.items():
            indicators = pattern_info["indicators"]
            primary_domain = pattern_info["primary_domain"]

            # Find relevant subcategories in the YAML structures for this pattern
            relevant_subcategories = []

            # Search through the domain structure to find matching subcategories
            if primary_domain in self.domain_structures:
                structure = self.domain_structures[primary_domain]
                relevant_subcategories = self._find_pattern_subcategories(
                    structure, pattern_info
                )

            self.dual_function_index[pattern_name] = {
                **pattern_info,
                "relevant_subcategories": relevant_subcategories,
            }

            logger.info(
                f"Built dual-function index for {pattern_name}: {len(relevant_subcategories)} relevant subcategories"
            )

    def _find_pattern_subcategories(self, structure, pattern_info):
        """Find subcategories in YAML structure that match the dual function pattern"""
        relevant_subcategories = []
        functions = pattern_info["functions"]

        def search_structure(node, path=[]):
            if isinstance(node, dict):
                for key, value in node.items():
                    current_path = path + [key]

                    # Check if this subcategory matches any of the functions
                    key_lower = key.lower()
                    for function in functions:
                        function_keywords = function.replace("_", " ").split()
                        if any(keyword in key_lower for keyword in function_keywords):
                            relevant_subcategories.append(
                                {
                                    "path": current_path,
                                    "function": function,
                                    "subcategory": key,
                                }
                            )

                    if value:
                        search_structure(value, current_path)

            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for function in functions:
                            function_keywords = function.replace("_", " ").split()
                            if any(
                                keyword in item_lower for keyword in function_keywords
                            ):
                                relevant_subcategories.append(
                                    {
                                        "path": path + [item],
                                        "function": function,
                                        "subcategory": item,
                                    }
                                )
                    elif isinstance(item, dict):
                        search_structure(item, path)

        search_structure(structure)
        return relevant_subcategories

    def _extract_keywords_from_structure(
        self, node, keywords: set, subcategories: set, level=0
    ):
        """Recursively extract keywords from YAML structure"""
        if level > 5:
            return

        if isinstance(node, dict):
            for key, value in node.items():
                key_tokens = self._tokenize_text(key)
                keywords.update(key_tokens)
                subcategories.add(key)

                if value:
                    self._extract_keywords_from_structure(
                        value, keywords, subcategories, level + 1
                    )

        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    item_tokens = self._tokenize_text(item)
                    keywords.update(item_tokens)
                    subcategories.add(item)
                elif isinstance(item, dict):
                    self._extract_keywords_from_structure(
                        item, keywords, subcategories, level + 1
                    )

    def _tokenize_text(self, text: str) -> set:
        """Tokenize text into meaningful keywords"""
        if not text:
            return set()

        text = text.lower()
        text = re.sub(r"[^\w\s-]", " ", text)

        tokens = set()
        words = re.split(r"[\s\-_/]+", text)

        for word in words:
            word = word.strip()
            if len(word) >= 3:
                tokens.add(word)

        return tokens

    def detect_dual_function_product(
        self, product_name: str, description: str = ""
    ) -> Optional[Dict]:
        """Detect if a product has dual functions based on patterns"""
        text = f"{product_name} {description}".lower()

        for pattern_name, pattern_info in self.dual_function_index.items():
            indicators = pattern_info["indicators"]

            # Check if any indicators match
            matches = []
            for indicator in indicators:
                if re.search(indicator, text):
                    matches.append(indicator)

            if matches:
                logger.info(
                    f"Dual-function pattern '{pattern_name}' detected for: {product_name} (matched: {matches})"
                )
                return {
                    "pattern_name": pattern_name,
                    "pattern_info": pattern_info,
                    "matched_indicators": matches,
                    "product_name": product_name,
                }

        return None

    def get_candidate_domains_enhanced(
        self, product_name: str, description: str
    ) -> List[Tuple[str, float, str]]:
        """Get candidate domains with enhanced scoring (same as before)"""
        text = f"{product_name} {description}".lower()
        product_tokens = self._tokenize_text(text)

        domain_scores = []

        for domain_key in self.domain_keywords.keys():
            domain_keywords = self.domain_keywords[domain_key]
            domain_subcategories = self.domain_subcategories[domain_key]

            keyword_overlap = product_tokens.intersection(domain_keywords)
            subcategory_overlap = set()

            for subcat in domain_subcategories:
                subcat_tokens = self._tokenize_text(subcat)
                if subcat_tokens.intersection(product_tokens):
                    subcategory_overlap.add(subcat)

            score = 0
            reasoning_parts = []

            if keyword_overlap:
                keyword_score = len(keyword_overlap) * 2
                score += keyword_score
                reasoning_parts.append(f"keyword matches: {list(keyword_overlap)[:3]}")

            if subcategory_overlap:
                subcat_score = len(subcategory_overlap) * 5
                score += subcat_score
                reasoning_parts.append(
                    f"subcategory matches: {list(subcategory_overlap)[:2]}"
                )

            # Domain-specific pattern matching
            domain_specific_score, domain_reasoning = (
                self._check_domain_specific_patterns(
                    product_name, description, domain_key
                )
            )
            score += domain_specific_score
            if domain_reasoning:
                reasoning_parts.append(domain_reasoning)

            if score > 0:
                reasoning = "; ".join(reasoning_parts)
                domain_scores.append((domain_key, score, reasoning))

        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return domain_scores[:6]

    def _check_domain_specific_patterns(
        self, product_name: str, description: str, domain_key: str
    ) -> Tuple[float, str]:
        """Check for domain-specific patterns with enhanced antibody detection"""
        text = f"{product_name} {description}".lower()

        patterns = {
            "Antibodies": [
                (r"\banti[- ]?[a-z0-9]+\b", 25, "anti-X pattern"),
                (r"\b(monoclonal|polyclonal)\b", 20, "antibody type"),
                (r"\b(primary|secondary)\s+antibody\b", 22, "antibody category"),
                (r"\bantibody\b", 15, "explicit antibody"),
                (r"\bigg?[gma]?\b", 10, "immunoglobulin"),
                (r"\b(hrp|fitc|pe|apc|alexa)\b", 12, "conjugation"),
                (r"\b(bcl|cd\d+|egfr|her2|p53|vegf)\b", 18, "known antibody targets"),
            ],
            "Cell_Biology": [
                (r"\b[a-z]+\s*\d+\s*cell\s*line\b", 20, "cell line pattern"),
                (r"\b(hela|mcf|a549|cho|hek|kyse)\b", 18, "known cell line"),
                (r"\bcell\s+(culture|viability|tracking)\b", 12, "cell techniques"),
                (r"\b(transfection|transformation)\b", 10, "cell manipulation"),
            ],
            "PCR": [
                (r"\b(q?pcr|thermocycler|thermal\s+cycler)\b", 15, "PCR equipment"),
                (r"\b(master\s+mix|polymerase|taq)\b", 12, "PCR reagents"),
                (r"\b(primer|probe)\b", 10, "PCR components"),
            ],
        }

        domain_patterns = patterns.get(domain_key, [])
        total_score = 0
        matched_reasons = []

        for pattern, score, reason in domain_patterns:
            if re.search(pattern, text):
                total_score += score
                matched_reasons.append(reason)

        # Special boost for antibodies
        if domain_key == "Antibodies" and re.search(r"\banti[- ]?[a-z0-9]+\b", text):
            total_score += 30
            matched_reasons.append("strong antibody indicator")

        # Penalty for non-antibody domains when clear antibody patterns exist
        if domain_key != "Antibodies" and re.search(
            r"\banti[- ]?[a-z0-9]+\s+antibody\b", text
        ):
            total_score -= 20

        reasoning = f"domain patterns: {matched_reasons}" if matched_reasons else ""
        return total_score, reasoning

    def get_focused_structure_for_prompt(
        self, domain_key: str, product_name: str = "", description: str = ""
    ) -> str:
        """Get focused, relevant structure for LLM prompt using enhanced master categories"""

        if domain_key not in self.domain_structures:
            return f"Domain '{domain_key}' not found"

        # Get enhanced master category info
        domain_info = self.master_config.get("domain_mapping", {}).get(domain_key, {})

        lines = [f"DOMAIN: {domain_key.replace('_', ' ')}"]
        lines.append(f"Description: {domain_info.get('description', '')}")
        lines.append("")

        # Show key classification paths from master categories
        key_paths = domain_info.get("key_classification_paths", [])
        if key_paths:
            lines.append("KEY CLASSIFICATION PATHS:")

            # Get relevant paths based on product keywords
            relevant_paths = self._get_relevant_classification_paths(
                key_paths, product_name, description
            )

            for path in relevant_paths[:8]:  # Show top 8 most relevant
                lines.append(f"- {path}")

            lines.append("")

        # Show classification hints
        hints = domain_info.get("classification_hints", [])
        if hints:
            lines.append("CLASSIFICATION GUIDANCE:")
            for hint in hints[:5]:  # Show top 5 hints
                lines.append(f"- {hint}")
            lines.append("")

        # Show typical products as examples
        typical_products = domain_info.get("typical_products", [])
        if typical_products:
            lines.append(f"EXAMPLE PRODUCTS: {', '.join(typical_products)}")
            lines.append("")

        lines.append("INSTRUCTIONS:")
        lines.append(
            "1. Use EXACT path names from the 'Key Classification Paths' above"
        )
        lines.append("2. Choose the most specific path that matches this product")
        lines.append("3. Follow the classification guidance for this domain")
        lines.append("4. Navigate as deep as possible (aim for 3-4 levels)")

        return "\n".join(lines)

    def _get_relevant_classification_paths(
        self, key_paths: List[str], product_name: str, description: str
    ) -> List[str]:
        """Filter and sort classification paths by relevance to the product"""

        text = f"{product_name} {description}".lower()
        product_tokens = self._tokenize_text(text)

        path_relevance = []

        for path in key_paths:
            # Calculate relevance score for this path
            path_tokens = self._tokenize_text(path.replace(" -> ", " "))
            relevance_score = len(product_tokens.intersection(path_tokens))

            # Boost score for exact matches
            for token in product_tokens:
                if token in path.lower():
                    relevance_score += 2

            path_relevance.append((path, relevance_score))

        # Sort by relevance score (high first), then alphabetically
        path_relevance.sort(key=lambda x: (-x[1], x[0]))

        # Return paths, prioritizing relevant ones but including some general ones
        relevant_paths = [path for path, score in path_relevance if score > 0]
        general_paths = [path for path, score in path_relevance if score == 0]

        # Combine: relevant paths first, then some general ones
        return relevant_paths + general_paths[: max(0, 8 - len(relevant_paths))]

    def get_master_category_guidance(self, domain_key: str) -> str:
        """Get enhanced guidance from master categories"""

        domain_info = self.master_config.get("domain_mapping", {}).get(domain_key, {})

        lines = []
        lines.append(f"DOMAIN: {domain_key.replace('_', ' ').upper()}")
        lines.append(f"Description: {domain_info.get('description', '')}")

        # Show key indicators
        keywords = domain_info.get("keywords", [])
        if keywords:
            lines.append(f"Key indicators: {', '.join(keywords[:8])}")

        # Show dual function indicators if present
        dual_indicators = domain_info.get("dual_function_indicators", [])
        if dual_indicators:
            lines.append(f"Dual function markers: {', '.join(dual_indicators)}")

        return "\n".join(lines)

    def validate_classification_path(
        self, domain_key: str, path_components: List[str]
    ) -> Tuple[bool, str]:
        """Enhanced validation that handles actual YAML structures better"""
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

        # Navigate through the structure
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
                # Direct match
                if component in current_node:
                    validated_path.append(component)
                    current_node = current_node[component]
                    found = True
                    logger.info(f"Direct match: '{component}'")
                else:
                    # Fuzzy matching - try partial matches
                    matches = []
                    for key in current_node.keys():
                        if (
                            component.lower() in key.lower()
                            or key.lower() in component.lower()
                        ):
                            matches.append(key)

                    if matches:
                        best_match = matches[0]
                        validated_path.append(best_match)
                        current_node = current_node[best_match]
                        found = True
                        logger.info(f"Fuzzy matched '{component}' to '{best_match}'")

                # Navigate deeper if we found a match
                if found and isinstance(current_node, dict):
                    if "subsubcategories" in current_node:
                        current_node = current_node["subsubcategories"]
                        logger.info(f"Navigated into subsubcategories level")

            elif isinstance(current_node, list):
                # Find in list - try exact and partial matches
                exact_matches = [
                    item
                    for item in current_node
                    if isinstance(item, str) and item == component
                ]
                if exact_matches:
                    validated_path.append(exact_matches[0])
                    found = True
                    logger.info(f"Exact match in list: '{exact_matches[0]}'")
                else:
                    # Try partial matches
                    partial_matches = [
                        item
                        for item in current_node
                        if isinstance(item, str)
                        and (
                            component.lower() in item.lower()
                            or item.lower() in component.lower()
                        )
                    ]
                    if partial_matches:
                        validated_path.append(partial_matches[0])
                        found = True
                        logger.info(
                            f"Partial match in list: '{component}' matched to '{partial_matches[0]}'"
                        )

            if not found:
                # If we couldn't find this component, it's still okay if we got some depth
                if len(validated_path) > 1:  # At least domain + one level
                    logger.info(
                        f"Stopping validation at: {' -> '.join(validated_path)} (couldn't find '{component}')"
                    )
                    break
                else:
                    return (
                        False,
                        f"Invalid path: {' -> '.join(validated_path)} -> '{component}'",
                    )

        final_path = " -> ".join(validated_path)
        logger.info(f"Successfully validated path: {final_path}")
        return True, final_path


class EnhancedDualFunctionClassifier:
    """Enhanced classifier with dual-function multi-classification capabilities"""

    def __init__(
        self,
        category_system: EnhancedDualFunctionCategorySystem,
        tag_system: EnhancedTagSystem,
    ):
        self.category_system = category_system
        self.tag_system = tag_system
        self.logger = logging.getLogger(__name__)

    def classify_product(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Enhanced classification with dual-function detection"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CLASSIFYING: '{product_name}'")
        self.logger.info(f"{'='*60}")

        # Step 1: Check for dual-function patterns
        dual_function_info = self.category_system.detect_dual_function_product(
            product_name, description
        )

        if dual_function_info:
            # DUAL-CLASSIFICATION: Product has dual functions
            return self._classify_dual_function_product(
                product_name, description, dual_function_info
            )
        else:
            # SINGLE-CLASSIFICATION: Regular product (MOST PRODUCTS)
            return self._classify_single_function_product(product_name, description)

    def _classify_dual_function_product(
        self, product_name: str, description: str, dual_function_info: Dict
    ) -> Dict[str, Any]:
        """Classify product with dual functions"""

        pattern_info = dual_function_info["pattern_info"]
        pattern_name = dual_function_info["pattern_name"]

        self.logger.info(f"Processing dual-function product: {pattern_name}")

        # Get candidate domains
        candidates = self.category_system.get_candidate_domains_enhanced(
            product_name, description
        )

        # Classify from each functional perspective
        classifications = []
        total_tokens = 0

        for i, context in enumerate(pattern_info["classification_contexts"]):
            function_type = pattern_info["functions"][i]

            self.logger.info(f"Classifying as: {function_type}")

            classification = self._analyze_functional_context(
                product_name,
                description,
                context,
                function_type,
                pattern_info["primary_domain"],
                is_primary=(i == 0),
            )

            if classification:
                classifications.append(classification)
                total_tokens += classification.get("token_usage", 0)

        # Validate and format results
        if len(classifications) >= 2:
            # Compile and validate tags from all classifications
            all_tags = self._compile_and_validate_tags(classifications)

            return {
                "classifications": classifications,
                "primary_classification": classifications[0],
                "secondary_classification": classifications[1],
                "is_dual_function": True,
                "dual_function_pattern": pattern_name,
                "dual_function_reasoning": pattern_info["reasoning"],
                "classification_count": len(classifications),
                "candidate_domains": [(d, s) for d, s, r in candidates],
                "total_token_usage": total_tokens,
                **all_tags,
            }
        else:
            # Fall back to single classification if dual classification failed
            return self._classify_single_function_product(product_name, description)

    def _analyze_functional_context(
        self,
        product_name: str,
        description: str,
        context: str,
        function_type: str,
        primary_domain: str,
        is_primary: bool,
    ) -> Optional[Dict]:
        """Analyze product from a specific functional context"""

        # Get focused structure and guidance
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            primary_domain, product_name, description
        )
        master_guidance = self.category_system.get_master_category_guidance(
            primary_domain
        )

        # Get tag lists for prompt
        tag_lists = self.tag_system.get_tag_lists_for_prompt()

        # Build context-specific prompt with focused information
        prompt = f"""You are a life science product classification expert.

{master_guidance}

TASK: Classify this product focusing specifically on its role as: {context}

This product has dual functions, and you are analyzing it from the perspective of: {function_type}

{focused_structure}

Product: "{product_name}"
Description: "{description}"

FOCUS: Classify based on its function as {context}

CRITICAL RULES:
1. Use EXACT names from the paths shown above
2. Navigate as deep as possible (aim for 3-4 levels)
3. Choose the most specific subcategory that represents this functional role

MANDATORY TAG ASSIGNMENT:
Select 2-3 appropriate tags from each category:

TECHNIQUE TAGS: {tag_lists['technique_tags']}
RESEARCH TAGS: {tag_lists['research_tags']} 
FUNCTIONAL TAGS: {tag_lists['functional_tags']}

Respond with JSON only:
{{
    "domain_fit_score": 85,
    "belongs_in_domain": true,
    "subcategory": "exact_subcategory_name",
    "subsubcategory": "exact_subsubcategory_name", 
    "subsubsubcategory": "exact_3rd_level_name_or_null",
    "confidence": "High",
    "functional_context": "{function_type}",
    "reasoning": "detailed explanation focusing on this specific functional role",
    "technique_tags": ["tag1", "tag2"],
    "research_tags": ["tag1", "tag2"],
    "functional_tags": ["tag1", "tag2"],
    "specimen_tags": ["tag1"]
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)
            result = json.loads(cleaned)

            # Validate tags
            result["technique_tags"] = self.tag_system.validate_tags(
                result.get("technique_tags", []), "technique"
            )
            result["research_tags"] = self.tag_system.validate_tags(
                result.get("research_tags", []), "research"
            )
            result["functional_tags"] = self.tag_system.validate_tags(
                result.get("functional_tags", []), "functional"
            )
            result["specimen_tags"] = self.tag_system.validate_tags(
                result.get("specimen_tags", []), "specimen"
            )

            # Validate the classification path
            path_components = [
                result.get("subcategory", ""),
                result.get("subsubcategory", ""),
                result.get("subsubsubcategory", ""),
            ]
            path_components = [comp for comp in path_components if comp]

            is_valid, validated_path = (
                self.category_system.validate_classification_path(
                    primary_domain, path_components
                )
            )

            # Enhance result
            result.update(
                {
                    "domain": primary_domain,
                    "is_valid_path": is_valid,
                    "validated_path": validated_path,
                    "is_primary": is_primary,
                    "functional_context": function_type,
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            self.logger.info(
                f"Functional context analysis complete: {function_type}, fit_score={result.get('domain_fit_score', 0)}"
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Error analyzing functional context {function_type}: {e}"
            )
            return None

    def _classify_single_function_product(
        self, product_name: str, description: str
    ) -> Dict[str, Any]:
        """Single classification for regular products - MUST classify every product"""

        self.logger.info(f"Processing single-function product: {product_name}")

        # Get candidate domains
        candidates = self.category_system.get_candidate_domains_enhanced(
            product_name, description
        )

        if not candidates:
            self.logger.warning(f"No candidate domains found for: {product_name}")
            return self._create_fallback_result(product_name)

        # Try multiple candidate domains until we get a successful classification
        best_analysis = None
        total_tokens = 0

        for domain_key, score, reasoning in candidates[:3]:  # Try top 3 candidates
            self.logger.info(f"Analyzing domain: {domain_key} (score: {score})")

            analysis = self._analyze_domain_fit(product_name, description, domain_key)
            total_tokens += analysis.get("token_usage", 0) if analysis else 0

            if analysis:
                # Accept classification if:
                # 1. LLM says it belongs in domain, OR
                # 2. Has decent fit score (≥50), OR
                # 3. Has valid path regardless of fit score
                belongs = analysis.get("belongs_in_domain", False)
                fit_score = analysis.get("domain_fit_score", 0)
                valid_path = analysis.get("is_valid_path", False)

                if belongs or fit_score >= 50 or valid_path:
                    best_analysis = analysis
                    self.logger.info(
                        f"Accepted classification in {domain_key}: belongs={belongs}, fit={fit_score}, valid_path={valid_path}"
                    )
                    break
                else:
                    self.logger.info(
                        f"Rejected classification in {domain_key}: belongs={belongs}, fit={fit_score}, valid_path={valid_path}"
                    )

        if not best_analysis:
            # If no domain worked, use the first candidate with lowest threshold
            self.logger.warning(
                f"No acceptable classification found, using fallback for: {product_name}"
            )
            if candidates:
                domain_key, score, reasoning = candidates[0]
                analysis = self._analyze_domain_fit(
                    product_name, description, domain_key
                )
                if analysis:
                    # Force acceptance with lower standards
                    analysis["belongs_in_domain"] = True
                    analysis["confidence"] = "Low"
                    best_analysis = analysis
                    total_tokens += analysis.get("token_usage", 0)

        if best_analysis:
            # Compile and validate tags
            all_tags = self._compile_and_validate_tags([best_analysis])

            return {
                "classifications": [best_analysis],
                "primary_classification": best_analysis,
                "secondary_classification": None,
                "is_dual_function": False,
                "dual_function_pattern": "",
                "dual_function_reasoning": "",
                "classification_count": 1,
                "candidate_domains": [(d, s) for d, s, r in candidates],
                "total_token_usage": total_tokens,
                **all_tags,
            }
        else:
            # Ultimate fallback
            self.logger.error(f"Complete classification failure for: {product_name}")
            return self._create_fallback_result(product_name)

    def _analyze_domain_fit(
        self, product_name: str, description: str, domain_key: str
    ) -> Dict[str, Any]:
        """Analyze how well a product fits within a specific domain"""

        # Get focused structure and guidance instead of full YAML dump
        focused_structure = self.category_system.get_focused_structure_for_prompt(
            domain_key, product_name, description
        )
        master_guidance = self.category_system.get_master_category_guidance(domain_key)

        # Get tag lists for prompt
        tag_lists = self.tag_system.get_tag_lists_for_prompt()

        prompt = f"""You are a life science product classification expert.

{master_guidance}

TASK: Analyze if this product belongs in this domain and find the best classification path.

{focused_structure}

Product: "{product_name}"
Description: "{description}"

ANALYSIS REQUIREMENTS:
1. Domain Fit Score (0-100): How well does this product fit in this domain?
2. Best Classification Path: Use exact names from the paths shown above
3. Confidence Level: High/Medium/Low
4. Navigate as deep as possible (aim for 3-4 levels)

CRITICAL RULES:
- Use EXACT names from the relevant subcategories shown above
- Choose the most specific path that matches this product
- If uncertain about deeper levels, stop at a confident level

MANDATORY TAG ASSIGNMENT:
Select 2-3 appropriate tags from each category:

TECHNIQUE TAGS: {tag_lists['technique_tags']}
RESEARCH TAGS: {tag_lists['research_tags']}
FUNCTIONAL TAGS: {tag_lists['functional_tags']} 

Respond with JSON only:
{{
    "domain_fit_score": 85,
    "belongs_in_domain": true,
    "subcategory": "exact_subcategory_name",
    "subsubcategory": "exact_subsubcategory_name", 
    "subsubsubcategory": "exact_3rd_level_name_or_null",
    "confidence": "High",
    "reasoning": "detailed explanation of why this classification makes sense",
    "alternative_domains": ["Domain1", "Domain2"],
    "technique_tags": ["tag1", "tag2"],
    "research_tags": ["tag1", "tag2"],
    "functional_tags": ["tag1", "tag2"],
    "specimen_tags": ["tag1"]
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
            )

            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)

            self.logger.info(f"LLM Response for {product_name}: {cleaned[:200]}...")

            result = json.loads(cleaned)

            # Validate tags
            result["technique_tags"] = self.tag_system.validate_tags(
                result.get("technique_tags", []), "technique"
            )
            result["research_tags"] = self.tag_system.validate_tags(
                result.get("research_tags", []), "research"
            )
            result["functional_tags"] = self.tag_system.validate_tags(
                result.get("functional_tags", []), "functional"
            )
            result["specimen_tags"] = self.tag_system.validate_tags(
                result.get("specimen_tags", []), "specimen"
            )

            # Validate the classification path
            path_components = [
                result.get("subcategory", ""),
                result.get("subsubcategory", ""),
                result.get("subsubsubcategory", ""),
            ]
            path_components = [comp for comp in path_components if comp]

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
                f"Domain analysis result: fit_score={result.get('domain_fit_score', 0)}, valid_path={is_valid}"
            )

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error for '{product_name}': {text}")
            self.logger.error(f"Error details: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing domain {domain_key}: {e}")
            return None

    def _compile_and_validate_tags(self, classifications: List[Dict]) -> Dict[str, Any]:
        """Compile and validate tags from all classifications"""
        all_technique_tags = []
        all_research_tags = []
        all_functional_tags = []
        all_specimen_tags = []

        for classification in classifications:
            all_technique_tags.extend(classification.get("technique_tags", []))
            all_research_tags.extend(classification.get("research_tags", []))
            all_functional_tags.extend(classification.get("functional_tags", []))
            all_specimen_tags.extend(classification.get("specimen_tags", []))

        # Remove duplicates while preserving order
        technique_tags = list(dict.fromkeys(all_technique_tags))
        research_tags = list(dict.fromkeys(all_research_tags))
        functional_tags = list(dict.fromkeys(all_functional_tags))
        specimen_tags = list(dict.fromkeys(all_specimen_tags))

        # Create combined tag list
        all_tags = technique_tags + research_tags + functional_tags + specimen_tags

        return {
            "technique_tags": technique_tags,
            "research_tags": research_tags,
            "functional_tags": functional_tags,
            "specimen_tags": specimen_tags,
            "all_tags": all_tags,
            "tag_count": len(all_tags),
        }

    def _create_fallback_result(self, product_name: str) -> Dict[str, Any]:
        """Create fallback result - ALWAYS provides a classification"""

        self.logger.warning(f"Using fallback classification for: {product_name}")

        fallback_tags = {
            "technique_tags": [],
            "research_tags": ["Cell Biology"],
            "functional_tags": ["Research Use Only"],
            "specimen_tags": [],
            "all_tags": ["Cell Biology", "Research Use Only"],
            "tag_count": 2,
        }

        fallback_classification = {
            "domain": "Other",
            "subcategory": "Unclassified",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "domain_fit_score": 10,
            "reasoning": "Could not determine specific classification",
            "is_valid_path": False,
            "validated_path": "Other -> Unclassified",
            "is_primary": True,
            "functional_context": "",
            **fallback_tags,
        }

        return {
            "classifications": [fallback_classification],
            "primary_classification": fallback_classification,
            "secondary_classification": None,
            "is_dual_function": False,
            "dual_function_pattern": "",
            "dual_function_reasoning": "",
            "classification_count": 1,
            "candidate_domains": [],
            "total_token_usage": 0,
            **fallback_tags,
        }

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def test_dual_function_classification():
    """Test the dual-function classification system"""

    # Initialize systems
    category_system = EnhancedDualFunctionCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = EnhancedDualFunctionClassifier(category_system, tag_system)

    # Test cases with expected dual functions
    test_cases = [
        {
            "name": "anti-bcl-2 antibody",
            "description": "Monoclonal antibody against Bcl-2 protein for cancer and apoptosis research",
            "expected_dual_function": True,
            "expected_pattern": "cancer_apoptosis_markers",
        },
        {
            "name": "DAPI nucleic acid stain",
            "description": "Fluorescent stain for nuclear DNA visualization",
            "expected_dual_function": True,
            "expected_pattern": "nuclear_dna_stains",
        },
        {
            "name": "TNF-alpha recombinant protein",
            "description": "Recombinant tumor necrosis factor alpha cytokine",
            "expected_dual_function": True,
            "expected_pattern": "cytokine_growth_factors",
        },
        {
            "name": "trypsin solution",
            "description": "Enzyme for protein digestion and cell dissociation",
            "expected_dual_function": True,
            "expected_pattern": "protease_dissociation",
        },
        {
            "name": "qPCR master mix",
            "description": "Ready-to-use mix for quantitative PCR",
            "expected_dual_function": False,
            "expected_pattern": None,
        },
    ]

    print("=" * 80)
    print("TESTING DUAL-FUNCTION CLASSIFICATION SYSTEM")
    print("=" * 80)

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTEST {i}: {test_case['name']}")
        print("-" * 50)

        # Test dual-function detection
        dual_function_info = category_system.detect_dual_function_product(
            test_case["name"], test_case["description"]
        )

        detected_dual = dual_function_info is not None
        expected_dual = test_case["expected_dual_function"]

        print(f"Expected Dual Function: {expected_dual}")
        print(f"Detected Dual Function: {detected_dual}")

        if detected_dual == expected_dual:
            print("✓ Dual Function Detection: CORRECT")

            if detected_dual and test_case["expected_pattern"]:
                detected_pattern = dual_function_info["pattern_name"]
                expected_pattern = test_case["expected_pattern"]
                pattern_match = detected_pattern == expected_pattern
                print(f"Expected Pattern: {expected_pattern}")
                print(f"Detected Pattern: {detected_pattern}")
                print(f"✓ Pattern Match: {'CORRECT' if pattern_match else 'INCORRECT'}")
        else:
            print("✗ Dual Function Detection: INCORRECT")

        # Full classification
        result = classifier.classify_product(
            test_case["name"], test_case["description"]
        )

        classifications = result.get("classifications", [])
        is_dual_function = result.get("is_dual_function", False)

        print(f"\nClassification Results:")
        print(f"  Is Dual Function: {is_dual_function}")
        print(f"  Classification Count: {len(classifications)}")

        if classifications:
            primary = classifications[0] if classifications else {}
            print(
                f"  Primary: {primary.get('domain', '')} -> {primary.get('validated_path', '')}"
            )
            print(f"  Primary Context: {primary.get('functional_context', 'N/A')}")

            if len(classifications) > 1:
                secondary = classifications[1]
                print(
                    f"  Secondary: {secondary.get('domain', '')} -> {secondary.get('validated_path', '')}"
                )
                print(
                    f"  Secondary Context: {secondary.get('functional_context', 'N/A')}"
                )

        print(
            f"  Dual Function Reasoning: {result.get('dual_function_reasoning', 'N/A')}"
        )
        print(f"  Token Usage: {result.get('total_token_usage', 0)}")

        # Show tags
        print(f"  Technique Tags: {result.get('technique_tags', [])}")
        print(f"  Research Tags: {result.get('research_tags', [])}")
        print(f"  Functional Tags: {result.get('functional_tags', [])}")
        print(f"  Total Tags: {result.get('tag_count', 0)}")

        # Determine success
        detection_correct = detected_dual == expected_dual
        if expected_dual and detected_dual:
            # For dual function products, check if we got multiple classifications
            classification_correct = len(classifications) >= 2
        else:
            # For single function products, check if we got single classification
            classification_correct = len(classifications) == 1

        if detection_correct and classification_correct:
            success_count += 1
            print("✅ OVERALL: SUCCESS")
        else:
            print("❌ OVERALL: FAILED")

    print(f"\n{'='*80}")
    print(
        f"DUAL-FUNCTION TEST RESULTS: {success_count}/{len(test_cases)} passed ({success_count/len(test_cases)*100:.1f}%)"
    )
    print(f"{'='*80}")

    return success_count >= 4


def process_validation_sample():
    """Process validation sample with dual-function enhanced system"""
    logger.info("Starting dual-function enhanced validation sample processing...")

    # Initialize systems - FIX: Add missing tag_system
    category_system = EnhancedDualFunctionCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = EnhancedDualFunctionClassifier(category_system, tag_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        sample_size = min(100, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        validation_df = df.iloc[sample_indices].copy()

        # Add classification columns
        validation_df["primary_domain"] = ""
        validation_df["primary_subcategory"] = ""
        validation_df["primary_subsubcategory"] = ""
        validation_df["primary_subsubsubcategory"] = ""
        validation_df["primary_confidence"] = ""
        validation_df["primary_fit_score"] = 0
        validation_df["primary_path_valid"] = False
        validation_df["primary_functional_context"] = ""
        validation_df["validated_path"] = ""

        # Secondary classification columns
        validation_df["secondary_domain"] = ""
        validation_df["secondary_subcategory"] = ""
        validation_df["secondary_subsubcategory"] = ""
        validation_df["secondary_subsubsubcategory"] = ""
        validation_df["secondary_confidence"] = ""
        validation_df["secondary_fit_score"] = 0
        validation_df["secondary_functional_context"] = ""
        validation_df["secondary_path_valid"] = False

        # Dual function columns
        validation_df["is_dual_function"] = False
        validation_df["dual_function_pattern"] = ""
        validation_df["dual_function_reasoning"] = ""
        validation_df["classification_count"] = 0
        validation_df["total_token_usage"] = 0

        logger.info(f"Processing {len(validation_df)} products...")

        for idx in tqdm(
            validation_df.index, desc="Dual-Function Enhanced Classification"
        ):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            # Perform classification
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
                validation_df.at[idx, "primary_functional_context"] = primary.get(
                    "functional_context", ""
                )
                validation_df.at[idx, "validated_path"] = primary.get(
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
                validation_df.at[idx, "secondary_functional_context"] = secondary.get(
                    "functional_context", ""
                )
                validation_df.at[idx, "secondary_path_valid"] = secondary.get(
                    "is_valid_path", False
                )

            # Dual function metadata
            validation_df.at[idx, "is_dual_function"] = result.get(
                "is_dual_function", False
            )
            validation_df.at[idx, "dual_function_pattern"] = result.get(
                "dual_function_pattern", ""
            )
            validation_df.at[idx, "dual_function_reasoning"] = result.get(
                "dual_function_reasoning", ""
            )
            validation_df.at[idx, "classification_count"] = result.get(
                "classification_count", 0
            )
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )

        # Save results
        validation_df.to_csv(VALIDATION_CSV, index=False)
        logger.info(
            f"Dual-function enhanced validation sample saved to {VALIDATION_CSV}"
        )

        # Generate report
        generate_dual_function_validation_report(validation_df)

        return validation_df

    except Exception as e:
        logger.error(f"Error in validation processing: {e}")
        raise


def generate_dual_function_validation_report(validation_df: pd.DataFrame):
    """Generate dual-function validation report"""
    print("\n" + "=" * 80)
    print("DUAL-FUNCTION ENHANCED CLASSIFICATION VALIDATION REPORT")
    print("=" * 80)

    total_products = len(validation_df)
    classified_products = len(
        validation_df[
            validation_df["primary_domain"].notna()
            & (validation_df["primary_domain"] != "")
            & (validation_df["primary_domain"] != "Other")
        ]
    )

    dual_function_products = len(
        validation_df[validation_df["is_dual_function"] == True]
    )

    print(f"Total products validated: {total_products}")
    print(
        f"Successfully classified: {classified_products} ({classified_products/total_products*100:.1f}%)"
    )
    print(
        f"Dual-function products: {dual_function_products} ({dual_function_products/total_products*100:.1f}%)"
    )

    # Domain distribution
    print(f"\n{'PRIMARY DOMAIN DISTRIBUTION':-^60}")
    domain_counts = validation_df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(10).items():
        print(f"  {domain:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Dual function analysis
    if dual_function_products > 0:
        print(f"\n{'DUAL-FUNCTION ANALYSIS':-^60}")
        pattern_counts = validation_df[validation_df["is_dual_function"] == True][
            "dual_function_pattern"
        ].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern:<35} {count:>5}")

        # Show some examples
        print(f"\n{'DUAL-FUNCTION EXAMPLES':-^60}")
        dual_examples = validation_df[validation_df["is_dual_function"] == True].head(3)
        for idx, row in dual_examples.iterrows():
            print(f"  {row['Name'][:40]:<40}")
            print(f"    Primary: {row['primary_functional_context']}")
            print(f"    Secondary: {row['secondary_functional_context']}")
            print(f"    Reasoning: {row['dual_function_reasoning'][:60]}...")
            print()

    # Confidence distribution
    print(f"\n{'CONFIDENCE DISTRIBUTION':-^60}")
    confidence_counts = validation_df["primary_confidence"].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Fit score analysis
    fit_scores = validation_df["primary_fit_score"].dropna()
    if len(fit_scores) > 0:
        print(f"\n{'FIT SCORE ANALYSIS':-^60}")
        print(f"  Average fit score: {fit_scores.mean():.1f}")
        print(
            f"  High fit scores (≥80): {len(fit_scores[fit_scores >= 80])} ({len(fit_scores[fit_scores >= 80])/len(fit_scores)*100:.1f}%)"
        )

    # Path validity
    valid_paths = len(validation_df[validation_df["primary_path_valid"] == True])
    print(f"\n{'PATH VALIDITY':-^60}")
    print(
        f"  Valid primary paths: {valid_paths} ({valid_paths/total_products*100:.1f}%)"
    )

    if dual_function_products > 0:
        valid_secondary_paths = len(
            validation_df[validation_df["secondary_path_valid"] == True]
        )
        print(
            f"  Valid secondary paths: {valid_secondary_paths} ({valid_secondary_paths/dual_function_products*100:.1f}% of dual-function)"
        )

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
    print("YAML-BASED ENHANCED DUAL-FUNCTION CLASSIFICATION SYSTEM")
    print("=" * 80)

    print(f"Looking for master categories file: {MASTER_CATEGORIES_FILE}")
    print(f"Looking for YAML files in: {YAML_DIRECTORY}")

    try:
        # Test dual-function classification
        print("\n1. Testing dual-function classification...")
        test_success = test_dual_function_classification()

        if test_success:
            print("\n" + "=" * 80)
            user_input = input(
                "Tests passed! Proceed with validation sample processing? (y/n): "
            )

            if user_input.lower() == "y":
                validation_df = process_validation_sample()
                print("\n" + "=" * 80)
                print("🎉 DUAL-FUNCTION ENHANCED VALIDATION COMPLETE! 🎉")
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
        print(f"ERROR: {e}")


def process_full_dataset():
    """Process the full dataset with dual-function enhanced classification"""
    logger.info("Starting full dataset processing with dual-function enhancement...")

    # Initialize systems - FIX: Add missing tag_system
    category_system = EnhancedDualFunctionCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = EnhancedDualFunctionClassifier(category_system, tag_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df)} products from {INPUT_CSV}")

        # Add all classification columns
        new_columns = [
            "primary_domain",
            "primary_subcategory",
            "primary_subsubcategory",
            "primary_subsubsubcategory",
            "primary_confidence",
            "primary_fit_score",
            "primary_path_valid",
            "primary_functional_context",
            "secondary_domain",
            "secondary_subcategory",
            "secondary_subsubcategory",
            "secondary_subsubsubcategory",
            "secondary_confidence",
            "secondary_fit_score",
            "secondary_path_valid",
            "secondary_functional_context",
            "is_dual_function",
            "dual_function_pattern",
            "dual_function_reasoning",
            "classification_count",
            "total_token_usage",
            "validated_path",
        ]

        for col in new_columns:
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

                    # Store all results (same logic as validation)
                    classifications = result.get("classifications", [])

                    if classifications:
                        primary = classifications[0]
                        df.at[idx, "primary_domain"] = primary.get("domain", "")
                        df.at[idx, "primary_subcategory"] = primary.get(
                            "subcategory", ""
                        )
                        df.at[idx, "primary_subsubcategory"] = primary.get(
                            "subsubcategory", ""
                        )
                        df.at[idx, "primary_subsubsubcategory"] = primary.get(
                            "subsubsubcategory", ""
                        )
                        df.at[idx, "primary_confidence"] = primary.get("confidence", "")
                        df.at[idx, "primary_fit_score"] = primary.get(
                            "domain_fit_score", 0
                        )
                        df.at[idx, "primary_path_valid"] = primary.get(
                            "is_valid_path", False
                        )
                        df.at[idx, "primary_functional_context"] = primary.get(
                            "functional_context", ""
                        )
                        df.at[idx, "validated_path"] = primary.get("validated_path", "")

                    if len(classifications) > 1:
                        secondary = classifications[1]
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
                        df.at[idx, "secondary_functional_context"] = secondary.get(
                            "functional_context", ""
                        )
                        df.at[idx, "secondary_path_valid"] = secondary.get(
                            "is_valid_path", False
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
                    df.at[idx, "classification_count"] = result.get(
                        "classification_count", 0
                    )
                    df.at[idx, "total_token_usage"] = result.get("total_token_usage", 0)

                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")
                    df.at[idx, "primary_domain"] = "Error"

            # Save progress after each batch
            df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Saved progress after batch {batch_num + 1}")

        logger.info(
            f"Full dual-function enhanced dataset processing complete. Results saved to {OUTPUT_CSV}"
        )

        # Generate final report
        generate_final_dual_function_report(df)

    except Exception as e:
        logger.error(f"Error in full dataset processing: {e}")
        raise


def generate_final_dual_function_report(df: pd.DataFrame):
    """Generate final dual-function processing report"""
    print("\n" + "=" * 80)
    print("FINAL DUAL-FUNCTION ENHANCED CLASSIFICATION REPORT")
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

    dual_function_products = len(df[df["is_dual_function"] == True])

    print(f"Total products processed: {total_products:,}")
    print(
        f"Successfully classified: {classified_products:,} ({classified_products/total_products*100:.1f}%)"
    )
    print(
        f"Dual-function products identified: {dual_function_products:,} ({dual_function_products/total_products*100:.1f}%)"
    )

    # Domain distribution
    print(f"\n{'DOMAIN DISTRIBUTION':-^60}")
    domain_counts = df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(15).items():
        print(f"  {domain:<35} {count:>7,} ({count/total_products*100:>5.1f}%)")

    # Dual function patterns
    if dual_function_products > 0:
        print(f"\n{'DUAL-FUNCTION PATTERNS':-^60}")
        pattern_counts = df[df["is_dual_function"] == True][
            "dual_function_pattern"
        ].value_counts()
        for pattern, count in pattern_counts.items():
            print(
                f"  {pattern:<35} {count:>7,} ({count/dual_function_products*100:>5.1f}%)"
            )

    # Quality metrics
    high_quality = len(
        df[
            (df["primary_confidence"] == "High")
            & (df["primary_path_valid"] == True)
            & (df["primary_fit_score"] >= 70)
        ]
    )

    print(f"\n{'QUALITY ANALYSIS':-^60}")
    print(
        f"  High quality classifications: {high_quality:,} ({high_quality/total_products*100:.1f}%)"
    )

    # Token usage
    total_tokens = df["total_token_usage"].sum()
    print(f"\n{'COST ANALYSIS':-^60}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${total_tokens * 0.00015 / 1000:.2f}")

    print(f"\n{'='*80}")
    print("🎉 DUAL-FUNCTION ENHANCED CLASSIFICATION SYSTEM COMPLETE! 🎉")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
