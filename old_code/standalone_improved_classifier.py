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

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Update these paths to match your setup
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_improved_classification.csv"
VALIDATION_CSV = "validation_improved_classification.csv"
CHECKPOINT_DIR = "checkpoints_improved"
CHECKPOINT_FREQ = 100
VALIDATION_SAMPLE_SIZE = 100

# YAML file configuration
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
    logger.error(f"OpenAI key not found in {env_path}")
    raise FileNotFoundError("OpenAI API key not found")


# Initialize OpenAI client
client = OpenAI(api_key=get_openai_key())

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED TAG SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedTagSystem:
    """Comprehensive cross-cutting tag system for life science products"""

    def __init__(self):
        # Core technique/application tags
        self.technique_tags = {
            # Antibody Techniques
            "Western Blot", "Immunohistochemistry", "Immunofluorescence", 
            "Flow Cytometry", "ELISA", "Immunoprecipitation", "ChIP", "ELISPOT",
            # Cell Analysis
            "Cell Culture", "Live Cell Imaging", "High Content Screening", 
            "Single Cell Analysis", "Cell Viability", "Apoptosis Analysis",
            "Cell Cycle Analysis", "Cell Tracking", "Cell Sorting", "Cell Counting",
            # Molecular Biology
            "PCR", "qPCR", "RT-PCR", "DNA Sequencing", "RNA Sequencing", 
            "Cloning", "Transfection", "Transformation", "Gene Expression",
            "In Situ Hybridization", "Northern Blot", "Southern Blot", 
            "CRISPR", "RNAi", "Mutagenesis",
            # Protein Analysis
            "Protein Purification", "Mass Spectrometry", "Chromatography", 
            "Electrophoresis", "Protein Crystallization", "Surface Plasmon Resonance",
            "BioLayer Interferometry", "Protein-Protein Interaction", 
            "Enzyme Assay", "Kinase Assay",
            # Imaging & Microscopy
            "Confocal Microscopy", "Fluorescence Microscopy", "Phase Contrast", 
            "DIC", "Super-Resolution Microscopy", "Electron Microscopy",
            "Light Sheet Microscopy", "TIRF Microscopy", "Multiphoton Microscopy",
            # Automation & High-Throughput
            "Liquid Handling", "Automated Workstation", "High-Throughput Screening",
            "Robotic System", "Plate Reader", "Automated Imaging"
        }

        # Research application areas
        self.research_application_tags = {
            # Basic Research Areas
            "Cell Biology", "Molecular Biology", "Biochemistry", "Immunology", 
            "Neuroscience", "Cancer Research", "Stem Cell Research", 
            "Developmental Biology", "Genetics", "Genomics", "Proteomics", 
            "Metabolomics", "Epigenetics", "Structural Biology",
            # Disease Areas
            "Oncology", "Neurodegenerative Disease", "Cardiovascular Disease", 
            "Autoimmune Disease", "Infectious Disease", "Metabolic Disease", 
            "Genetic Disorders", "Rare Disease",
            # Therapeutic Areas
            "Drug Discovery", "Biomarker Discovery", "Diagnostics", 
            "Companion Diagnostics", "Personalized Medicine", "Immunotherapy", 
            "Gene Therapy", "Cell Therapy",
            # Model Systems
            "Human Research", "Mouse Model", "Rat Model", "Cell Line", 
            "Primary Cells", "Organoid", "Tissue Culture", "In Vivo", 
            "In Vitro", "Ex Vivo",
            # Sample Types
            "Blood", "Plasma", "Serum", "Tissue", "FFPE", "Fresh Frozen", 
            "Cell Culture", "Bacterial", "Viral", "Plant", "Environmental"
        }

        # Functional/technical characteristics
        self.functional_tags = {
            # Performance Characteristics
            "High Sensitivity", "High Specificity", "Quantitative", "Qualitative", 
            "Semi-Quantitative", "Real-Time", "Endpoint", "Kinetic", 
            "High-Throughput", "Low-Throughput", "Multiplexed", "Single-Plex",
            "Automated", "Manual", "Walk-Away",
            # Product Format
            "Kit", "Reagent", "Instrument", "Consumable", "Software", "Service",
            "Ready-to-Use", "Requires Preparation", "Single-Use", "Reusable",
            # Quality & Compliance
            "Research Use Only", "Diagnostic Use", "GMP", "ISO Certified", 
            "FDA Approved", "CE Marked", "Validated", "Pre-Validated", 
            "Custom", "Standard",
            # Scale & Capacity
            "96-Well", "384-Well", "1536-Well", "Microplate", "Tube Format", 
            "Slide Format", "Benchtop", "Portable", "Handheld", "Large Scale", 
            "Small Scale", "Pilot Scale",
            # Technology Platform
            "Fluorescence", "Chemiluminescence", "Colorimetric", "Radiometric", 
            "Electrochemical", "Magnetic", "Acoustic", "Optical", "Digital", "Analog"
        }

        # Specimen/Sample tags
        self.specimen_tags = {
            "Human", "Mouse", "Rat", "Rabbit", "Bovine", "Porcine", "Canine", 
            "Feline", "Non-Human Primate", "Sheep", "Goat", "Horse", "Guinea Pig", 
            "Hamster", "Chicken", "Zebrafish", "C. elegans", "Drosophila", 
            "Yeast", "E. coli", "Plant", "Bacterial", "Viral", "Fungal"
        }

        self.all_tags = (
            self.technique_tags | self.research_application_tags | 
            self.functional_tags | self.specimen_tags
        )


# ═══════════════════════════════════════════════════════════════════════════════
# YAML CATEGORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class YAMLCategorySystem:
    """Manages YAML category files for product classification"""

    def __init__(self, yaml_directory: str = None):
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.category_files = {}
        self.master_categories = {}
        self.category_keywords = defaultdict(set)

        self.load_yaml_files()
        self.build_master_structure()
        self.build_keyword_index()

    def load_yaml_files(self):
        """Load all YAML category files"""
        if not os.path.exists(self.yaml_directory):
            logger.warning(f"YAML directory {self.yaml_directory} not found. Checking current directory...")
            self.yaml_directory = "."

        pattern = os.path.join(self.yaml_directory, "category_structure_*.yaml")
        yaml_files = glob.glob(pattern)

        if not yaml_files:
            yaml_files = glob.glob("category_structure_*.yaml")
            if yaml_files:
                self.yaml_directory = "."

        if not yaml_files:
            error_msg = f"No category_structure_*.yaml files found in {self.yaml_directory}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Found {len(yaml_files)} YAML category files in {self.yaml_directory}")

        for yaml_file in yaml_files:
            try:
                filename = os.path.basename(yaml_file)
                category_name = filename.replace("category_structure_", "").replace(".yaml", "")
                category_name = category_name.replace("_", " ").title()

                # Handle special cases
                category_name = category_name.replace("Bio Imaging", "Bioimaging/Microscopy")
                category_name = category_name.replace("Rnai Technology", "RNAi Technology")
                category_name = category_name.replace("Pcr", "PCR")

                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.category_files[category_name] = data.get("categories", {})

                logger.info(f"✓ Loaded: {category_name} from {filename}")

            except Exception as e:
                logger.error(f"✗ Error loading {yaml_file}: {e}")

        if not self.category_files:
            raise ValueError("No valid YAML files could be loaded!")

    def build_master_structure(self):
        """Build master category structure for domain overview"""
        self.master_categories = {}

        for category_name, structure in self.category_files.items():
            category_info = {}

            for main_cat, subcats in structure.items():
                if isinstance(subcats, dict) and subcats:
                    examples = list(subcats.keys())[:3]
                    category_info[main_cat] = {
                        "subcategory_examples": examples,
                        "total_subcategories": len(subcats),
                    }
                else:
                    category_info[main_cat] = {
                        "subcategory_examples": [],
                        "total_subcategories": 0,
                    }

            self.master_categories[category_name] = category_info

    def build_keyword_index(self):
        """Build keyword index for domain pre-filtering"""
        for category_name, structure in self.category_files.items():
            keywords = set()

            def extract_keywords(node, depth=0):
                if depth > 3:
                    return

                if isinstance(node, dict):
                    for key, value in node.items():
                        keywords.update(key.lower().split())
                        extract_keywords(value, depth + 1)
                elif isinstance(node, list):
                    for item in node:
                        if isinstance(item, str):
                            keywords.update(item.lower().split())
                        elif isinstance(item, dict):
                            extract_keywords(item, depth + 1)

            extract_keywords(structure)
            self.category_keywords[category_name] = keywords
            logger.info(f"Extracted {len(keywords)} keywords for {category_name}")

    def get_detailed_structure(self, category_name: str) -> str:
        """Get detailed YAML structure for specific category"""
        if category_name not in self.category_files:
            return f"Category '{category_name}' not found"

        structure = self.category_files[category_name]

        def format_structure(node, indent=0, max_depth=5):
            if indent >= max_depth:
                return ["  " * indent + "  • [more subcategories...]"]

            result = []
            prefix = "  " * indent

            if isinstance(node, dict):
                for key, value in node.items():
                    result.append(f"{prefix}- {key}")
                    if value and indent < max_depth - 1:
                        result.extend(format_structure(value, indent + 1, max_depth))
            elif isinstance(node, list):
                for i, item in enumerate(node[:10]):
                    if isinstance(item, str):
                        result.append(f"{prefix}  • {item}")
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            result.append(f"{prefix}  • {sub_key}")
                            if sub_value and indent < max_depth - 1:
                                result.extend(format_structure(sub_value, indent + 2, max_depth))
                            break
                if len(node) > 10:
                    result.append(f"{prefix}  • ... ({len(node)-10} more items)")

            return result

        lines = [f"COMPLETE STRUCTURE FOR: {category_name.upper()}"]
        lines.append("=" * 60)
        lines.extend(format_structure(structure))
        return "\n".join(lines)

    def validate_classification_path(self, category_name: str, path_components: List[str]) -> Tuple[bool, str]:
        """Validate a classification path within a specific category"""
        if category_name not in self.category_files:
            return False, f"Category '{category_name}' not found"

        structure = self.category_files[category_name]
        current_node = structure
        validated_path = [category_name]

        for component in path_components:
            if not component:
                continue

            if isinstance(current_node, dict):
                if component in current_node:
                    validated_path.append(component)
                    current_node = current_node[component]
                else:
                    # Try fuzzy matching
                    matches = [k for k in current_node.keys() if component.lower() in k.lower()]
                    if matches:
                        validated_path.append(matches[0])
                        current_node = current_node[matches[0]]
                    else:
                        return False, f"Invalid path: {' -> '.join(validated_path)} -> '{component}'"
            elif isinstance(current_node, list):
                matches = [item for item in current_node 
                          if isinstance(item, str) and component.lower() in item.lower()]
                if matches:
                    validated_path.append(matches[0])
                    break
                else:
                    return False, f"Invalid path: {' -> '.join(validated_path)} -> '{component}'"

        return True, " -> ".join(validated_path)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED TWO-STAGE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class ImprovedTwoStageClassifier:
    """Improved two-stage classification with full YAML structure access"""

    def __init__(self, category_system: YAMLCategorySystem, tag_system: EnhancedTagSystem):
        self.category_system = category_system
        self.tag_system = tag_system
        self.logger = logging.getLogger(__name__)

    def classify_product(self, product_name: str, description: str = "") -> Dict[str, Any]:
        """Complete improved two-stage classification process"""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CLASSIFYING: '{product_name}'")
        self.logger.info(f"{'='*60}")
        
        # STAGE 1: Domain identification only
        stage1_result = self.stage1_identify_domain(product_name, description)
        
        # STAGE 2: Full hierarchical classification within identified domain(s)
        classifications = []
        total_tokens = stage1_result.get("token_usage", 0)
        
        # Process primary domain
        primary_domain = stage1_result["primary_domain"]
        if primary_domain and primary_domain != "Other":
            stage2_result = self.stage2_full_classification(product_name, description, primary_domain)
            if self._is_classification_acceptable(stage2_result):
                classification = self._format_classification_result(stage2_result, is_primary=True)
                classifications.append(classification)
            total_tokens += stage2_result.get("token_usage", 0)
        
        # Process secondary domain if exists
        secondary_domain = stage1_result.get("secondary_domain")
        if secondary_domain and secondary_domain != primary_domain:
            stage2_result = self.stage2_full_classification(product_name, description, secondary_domain)
            if self._is_classification_acceptable(stage2_result):
                classification = self._format_classification_result(stage2_result, is_primary=False)
                classifications.append(classification)
            total_tokens += stage2_result.get("token_usage", 0)
        
        # Fallback if no acceptable classifications
        if not classifications:
            classifications.append(self._create_fallback_classification(primary_domain or "Other"))
        
        # Compile final result
        all_tags = []
        for classification in classifications:
            all_tags.extend(classification.get("technique_tags", []))
            all_tags.extend(classification.get("research_tags", []))
            all_tags.extend(classification.get("functional_tags", []))
        
        unique_tags = list(dict.fromkeys(all_tags))
        
        return {
            "classifications": classifications,
            "stage1_domains": {
                "primary": stage1_result["primary_domain"],
                "secondary": stage1_result.get("secondary_domain"),
            },
            "stage1_confidences": {
                "primary": stage1_result["primary_confidence"],
                "secondary": stage1_result.get("secondary_confidence"),
            },
            "all_tags": unique_tags,
            "tag_count": len(unique_tags),
            "classification_count": len(classifications),
            "total_token_usage": total_tokens,
            "stage1_reasoning": stage1_result.get("reasoning", ""),
        }

    def stage1_identify_domain(self, product_name: str, description: str = "") -> Dict[str, Any]:
        """STAGE 1: Clean domain identification with improved keyword matching"""
        
        self.logger.info(f"Stage 1: Domain identification for '{product_name}'")
        
        # Enhanced product name normalization
        normalized_name = self._normalize_product_name(product_name)
        self.logger.info(f"  Normalized name: '{normalized_name}' (from '{product_name}')")
        
        # Get candidate domains with improved keyword matching
        candidate_domains = self._get_candidate_domains(normalized_name, description)
        self.logger.info(f"  Candidate domains: {candidate_domains}")
        
        # Build domain overview for LLM
        domain_overview = self._build_domain_overview()
        
        # Focused domain identification prompt
        prompt = f"""You are a life science product classification expert.

TASK: Identify 1-2 broad domains for this product based on its primary function and application.

DOMAIN SELECTION RULES:
- Select PRIMARY domain where this product is most commonly used
- Only add SECONDARY domain if product has significant cross-domain applications
- Focus on the product's main PURPOSE, not just its components
- Be conservative - better 1 accurate domain than 2 questionable ones

SPECIFIC CLASSIFICATION GUIDELINES:
- Cell lines (e.g., HeLa, KYSE, A549, MCF-7) → Cell Biology
- cDNA synthesis kits/reagents → Cloning and Expression (NOT Molecular Biology)
- RNA isolation/purification kits → Nucleic Acid Purification  
- Antibodies for any application → Antibodies
- ELISA kits and immunoassays → Assay Kits
- PCR reagents/machines/primers → Polymerase Chain Reaction (PCR)
- Microscopes/imaging systems → Bioimaging/Microscopy
- Flow cytometers → Lab Equipment
- Western blot equipment → Lab Equipment
- Transfection reagents → RNAi Technology OR Cloning and Expression

{domain_overview}

Product: "{product_name}"
Normalized: "{normalized_name}"
Description: "{description}"

Respond with JSON only:
{{
    "primary_domain": "exact_domain_name",
    "secondary_domain": "exact_domain_name_or_null", 
    "primary_confidence": "High/Medium/Low",
    "secondary_confidence": "High/Medium/Low_or_null",
    "reasoning": "brief explanation focusing on product's primary function"
}}"""

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

            # Validate domains exist
            valid_domains = list(self.category_system.master_categories.keys())
            
            primary = result.get("primary_domain")
            secondary = result.get("secondary_domain")
            
            if primary not in valid_domains:
                primary = self._find_best_domain_match(primary, valid_domains)
                self.logger.warning(f"  Auto-corrected primary domain to: {primary}")
            
            if secondary and secondary not in valid_domains:
                secondary = self._find_best_domain_match(secondary, valid_domains)
                self.logger.warning(f"  Auto-corrected secondary domain to: {secondary}")
            
            # Apply improved filtering logic
            secondary = self._apply_domain_filtering(primary, secondary, 
                                                   result.get("primary_confidence"),
                                                   result.get("secondary_confidence"))
            
            self.logger.info(f"  Final domains: Primary={primary}, Secondary={secondary}")
            
            return {
                "primary_domain": primary,
                "secondary_domain": secondary,
                "primary_confidence": result.get("primary_confidence", "Medium"),
                "secondary_confidence": result.get("secondary_confidence") if secondary else None,
                "reasoning": result.get("reasoning", ""),
                "token_usage": response.usage.total_tokens if hasattr(response, 'usage') else 0,
            }

        except Exception as e:
            self.logger.error(f"Stage 1 classification error: {e}")
            return {
                "primary_domain": "Other",
                "secondary_domain": None,
                "primary_confidence": "Low",
                "secondary_confidence": None,
                "reasoning": "Domain identification failed",
                "token_usage": 0,
            }

    def stage2_full_classification(self, product_name: str, description: str, domain: str) -> Dict[str, Any]:
        """STAGE 2: Complete hierarchical classification with full YAML structure"""
        
        self.logger.info(f"Stage 2: Full classification for '{product_name}' in domain '{domain}'")
        
        # Get complete YAML structure for this domain
        complete_structure = self.category_system.get_detailed_structure(domain)
        self.logger.info(f"  Retrieved complete structure for {domain}")
        
        # Build comprehensive classification prompt
        prompt = self._build_comprehensive_classification_prompt(
            product_name, description, domain, complete_structure
        )
        
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
            
            self.logger.info(f"  Raw LLM result: {result}")
            
            # Comprehensive validation with full structure
            validated_result = self._validate_full_classification(domain, result)
            validated_result["token_usage"] = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return validated_result

        except Exception as e:
            self.logger.error(f"Stage 2 classification error: {e}")
            return self._create_fallback_stage2_result(domain)

    def _normalize_product_name(self, product_name: str) -> str:
        """Enhanced product name normalization"""
        
        name = product_name.lower().strip()
        
        # Brand name mappings
        brand_mappings = {
            'rneasy': 'rna isolation kit',
            'qiagen': '',
            'invitrogen': '',
            'thermo': '',
            'applied biosystems': '',
            'bio-rad': '',
            'bd': '',
            'millipore': '',
            'sigma': '',
            'affinityscript': 'cdna synthesis kit',
            'superscript': 'cdna synthesis kit',
            'oligo dt': 'cdna synthesis',
        }
        
        for brand, replacement in brand_mappings.items():
            if brand in name:
                name = name.replace(brand, replacement).strip()
        
        # Cell line mappings
        cell_line_patterns = {
            r'\bkyse\s*\d+\b': 'cell line',
            r'\bhela\b': 'cell line', 
            r'\ba549\b': 'cell line',
            r'\bmcf[_-]?7\b': 'cell line',
            r'\bht[_-]?29\b': 'cell line',
            r'\bu2os\b': 'cell line',
            r'\bhek\s*293\b': 'cell line',
            r'\bcho[_-]?k1\b': 'cell line',
        }
        
        for pattern, replacement in cell_line_patterns.items():
            name = re.sub(pattern, replacement, name)
        
        # Technical abbreviation expansions
        tech_mappings = {
            'cdna': 'complementary dna synthesis',
            'qpcr': 'quantitative pcr',
            'rtpcr': 'reverse transcription pcr',
            'elisa': 'enzyme linked immunosorbent assay',
            'facs': 'flow cytometry',
            'ihc': 'immunohistochemistry',
            'if': 'immunofluorescence',
            'wb': 'western blot',
            'ip': 'immunoprecipitation',
        }
        
        for abbrev, expansion in tech_mappings.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            name = re.sub(pattern, expansion, name)
        
        name = ' '.join(name.split())
        return name

    def _get_candidate_domains(self, product_name: str, description: str, max_domains: int = 5) -> List[str]:
        """Improved candidate domain selection with semantic weighting"""
        
        text = f"{product_name} {description}".lower()
        words = set(text.split())
        
        self.logger.info(f"  Product words: {sorted(list(words))}")
        
        domain_scores = {}
        
        # Enhanced scoring with semantic weights
        for domain_name, keywords in self.category_system.category_keywords.items():
            overlap = words.intersection(keywords)
            if overlap:
                score = 0
                for word in overlap:
                    # Give higher weight to specific technical terms
                    if len(word) > 8:  # Very specific terms
                        score += 5
                    elif len(word) > 6:  # Moderately specific
                        score += 3
                    elif word in ['synthesis', 'isolation', 'purification', 'assay', 'antibody', 'kit']:
                        score += 2
                    else:
                        score += 1
                
                domain_scores[domain_name] = score
                self.logger.info(f"  {domain_name}: {score} points from {overlap}")
        
        # Sort by score and return top candidates
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [domain for domain, score in sorted_domains[:max_domains]]
        
        return candidates

    def _build_domain_overview(self) -> str:
        """Build concise domain overview for Stage 1"""
        
        lines = ["AVAILABLE DOMAINS:"]
        
        for i, (domain_name, info) in enumerate(self.category_system.master_categories.items(), 1):
            lines.append(f"{i}. {domain_name}")
            # Add key examples
            examples = []
            for main_cat, details in list(info.items())[:2]:
                sub_examples = details.get("subcategory_examples", [])[:2]
                if sub_examples:
                    examples.extend(sub_examples)
            
            if examples:
                lines.append(f"   Examples: {', '.join(examples[:3])}...")
        
        return "\n".join(lines)

    def _build_comprehensive_classification_prompt(self, product_name: str, description: str, 
                                                 domain: str, structure: str) -> str:
        """Build comprehensive prompt with full YAML structure"""
        
        # Get tag examples
        technique_tags = list(sorted(self.tag_system.technique_tags))[:25]
        research_tags = list(sorted(self.tag_system.research_application_tags))[:20]
        functional_tags = list(sorted(self.tag_system.functional_tags))[:25]
        
        prompt = f"""You are a life science product classification expert.

TASK: Classify this product within the {domain} domain using the COMPLETE structure below.

CRITICAL RULES:
1. You MUST use EXACT names from the structure below - do not modify or create new names
2. Navigate as deep as possible in the hierarchy (aim for 4+ levels including domain)
3. Choose the MOST SPECIFIC category that accurately describes this product
4. You MUST assign tags from each category below

{structure}

MANDATORY TAG ASSIGNMENT - Select 1-3 tags from each category:

TECHNIQUE TAGS (how the product is used):
{', '.join(technique_tags)}

RESEARCH APPLICATION TAGS (what research it supports):
{', '.join(research_tags)}

FUNCTIONAL TAGS (what type of product it is):
{', '.join(functional_tags)}

Product: "{product_name}"
Description: "{description}"

CLASSIFICATION EXAMPLES:

For a cell line like "KYSE 150":
{{
    "category": "Cell Biology",
    "subcategory": "Primary Cells, Cell Lines and Microorganisms", 
    "subsubcategory": "Cell Lines",
    "subsubsubcategory": null,
    "confidence": "High",
    "technique_tags": ["Cell Culture", "Live Cell Imaging"],
    "research_tags": ["Cell Biology", "Cancer Research"], 
    "functional_tags": ["Research Use Only", "Cell Culture"]
}}

For a cDNA synthesis kit:
{{
    "category": "Cloning and Expression",
    "subcategory": "Library Construction",
    "subsubcategory": "cDNA Synthesis", 
    "subsubsubcategory": null,
    "confidence": "High",
    "technique_tags": ["PCR", "Gene Expression"],
    "research_tags": ["Molecular Biology", "Genomics"],
    "functional_tags": ["Kit", "Ready-to-Use"]
}}

Respond with JSON only, using EXACT names from the structure above:
{{
    "category": "exact_category_name",
    "subcategory": "exact_subcategory_name",
    "subsubcategory": "exact_subsubcategory_name",
    "subsubsubcategory": "exact_4th_level_name_or_null",
    "confidence": "High/Medium/Low", 
    "technique_tags": ["tag1", "tag2"],
    "research_tags": ["tag1"],
    "functional_tags": ["tag1", "tag2"]
}}"""

        return prompt

    def _validate_full_classification(self, domain: str, result: Dict) -> Dict[str, Any]:
        """Comprehensive validation using full YAML structure"""
        
        # Extract classification components
        category = result.get("category", "")
        subcategory = result.get("subcategory", "")
        subsubcategory = result.get("subsubcategory", "")
        subsubsubcategory = result.get("subsubsubcategory", "")
        
        self.logger.info(f"  Validating: {category} -> {subcategory} -> {subsubcategory} -> {subsubsubcategory}")
        
        # Use validation logic from category system
        path_components = [comp for comp in [category, subcategory, subsubcategory, subsubsubcategory] if comp]
        is_valid, validated_path = self.category_system.validate_classification_path(domain, path_components)
        
        # Validate tags
        technique_tags = [tag for tag in result.get("technique_tags", []) 
                         if tag in self.tag_system.technique_tags]
        research_tags = [tag for tag in result.get("research_tags", [])
                        if tag in self.tag_system.research_application_tags]
        functional_tags = [tag for tag in result.get("functional_tags", [])
                          if tag in self.tag_system.functional_tags]
        
        # Ensure minimum tags are assigned
        if not technique_tags:
            technique_tags = ["Cell Culture"]
        if not research_tags:
            research_tags = ["Cell Biology"]
        if not functional_tags:
            functional_tags = ["Research Use Only"]
        
        self.logger.info(f"  Validation result: is_valid={is_valid}, path='{validated_path}'")
        
        return {
            "domain": domain,
            "category": category,
            "subcategory": subcategory,
            "subsubcategory": subsubcategory,
            "subsubsubcategory": subsubsubcategory,
            "confidence": result.get("confidence", "Medium"),
            "technique_tags": technique_tags,
            "research_tags": research_tags,
            "functional_tags": functional_tags,
            "is_valid_path": is_valid,
            "validated_path": validated_path,
            "was_corrected": not is_valid,
        }

    def _apply_domain_filtering(self, primary: str, secondary: str, 
                               primary_conf: str, secondary_conf: str) -> Optional[str]:
        """Apply filtering logic for secondary domain"""
        
        if secondary and secondary != primary:
            if primary_conf == "High" and secondary_conf == "High":
                return secondary
            else:
                self.logger.info(f"  Filtered out secondary domain '{secondary}': confidence not high enough")
                return None
        
        return None

    def _is_classification_acceptable(self, result: Dict[str, Any]) -> bool:
        """Check if classification result is acceptable"""
        return (
            result.get("is_valid_path", False) or 
            result.get("confidence") in ["High", "Medium"]
        )

    def _format_classification_result(self, stage2_result: Dict, is_primary: bool) -> Dict[str, Any]:
        """Format Stage 2 result into final classification format"""
        return {
            "domain": stage2_result.get("domain", ""),
            "category": stage2_result.get("category", ""),
            "subcategory": stage2_result.get("subcategory", ""),
            "subsubcategory": stage2_result.get("subsubcategory", ""),
            "subsubsubcategory": stage2_result.get("subsubsubcategory", ""),
            "confidence": stage2_result.get("confidence", "Low"),
            "is_primary": is_primary,
            "is_valid_path": stage2_result.get("is_valid_path", False),
            "validated_path": stage2_result.get("validated_path", ""),
            "technique_tags": stage2_result.get("technique_tags", []),
            "research_tags": stage2_result.get("research_tags", []),
            "functional_tags": stage2_result.get("functional_tags", []),
        }

    def _create_fallback_classification(self, domain: str) -> Dict[str, Any]:
        """Create fallback classification when all else fails"""
        return {
            "domain": domain,
            "category": "Unclassified",
            "subcategory": "",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "is_primary": True,
            "is_valid_path": False,
            "validated_path": f"{domain} -> Unclassified",
            "technique_tags": ["Cell Culture"],
            "research_tags": ["Cell Biology"],
            "functional_tags": ["Research Use Only"],
        }

    def _create_fallback_stage2_result(self, domain: str) -> Dict[str, Any]:
        """Create fallback Stage 2 result"""
        return {
            "domain": domain,
            "category": "Unclassified",
            "subcategory": "",
            "subsubcategory": "",
            "subsubsubcategory": "",
            "confidence": "Low",
            "technique_tags": [],
            "research_tags": [],
            "functional_tags": [],
            "is_valid_path": False,
            "validated_path": f"{domain} -> Unclassified",
            "token_usage": 0,
        }

    def _find_best_domain_match(self, target: str, valid_domains: List[str]) -> str:
        """Find best matching domain from valid options"""
        if not target:
            return "Other"
        
        target_lower = target.lower()
        
        # Exact match
        for domain in valid_domains:
            if target_lower == domain.lower():
                return domain
        
        # Substring match
        for domain in valid_domains:
            if target_lower in domain.lower() or domain.lower() in target_lower:
                return domain
        
        return "Other"

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def process_validation_sample_improved():
    """Process validation sample with improved classification system"""
    logger.info("Starting improved validation sample processing...")

    # Initialize systems
    category_system = YAMLCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = ImprovedTwoStageClassifier(category_system, tag_system)

    # Create or load validation sample
    if os.path.exists(VALIDATION_CSV):
        response = input(f"Validation file {VALIDATION_CSV} exists. Load existing (1) or create new (2)? ")
        if response == "1":
            validation_df = pd.read_csv(VALIDATION_CSV)
            logger.info(f"Loaded existing validation sample with {len(validation_df)} products")
        else:
            validation_df = create_validation_sample_improved()
    else:
        validation_df = create_validation_sample_improved()

    # Process unclassified products
    unprocessed_mask = validation_df["primary_domain"].isna() | (validation_df["primary_domain"] == "")
    unprocessed_indices = validation_df[unprocessed_mask].index

    if len(unprocessed_indices) > 0:
        logger.info(f"Processing {len(unprocessed_indices)} products with improved classification...")

        for idx in tqdm(unprocessed_indices, desc="Improved Classification"):
            name = validation_df.at[idx, "Name"]
            description = validation_df.at[idx, "Description"] if "Description" in validation_df.columns else ""

            # Perform improved classification
            result = classifier.classify_product(name, description)

            # Store results
            classifications = result.get("classifications", [])

            # Primary classification
            if classifications:
                primary = classifications[0]
                validation_df.at[idx, "primary_domain"] = primary.get("domain", "")
                validation_df.at[idx, "primary_category"] = primary.get("category", "")
                validation_df.at[idx, "primary_subcategory"] = primary.get("subcategory", "")
                validation_df.at[idx, "primary_subsubcategory"] = primary.get("subsubcategory", "")
                validation_df.at[idx, "primary_subsubsubcategory"] = primary.get("subsubsubcategory", "")
                validation_df.at[idx, "primary_confidence"] = primary.get("confidence", "")
                validation_df.at[idx, "primary_path_valid"] = primary.get("is_valid_path", False)

            # Secondary classification
            if len(classifications) > 1:
                secondary = classifications[1]
                validation_df.at[idx, "secondary_domain"] = secondary.get("domain", "")
                validation_df.at[idx, "secondary_category"] = secondary.get("category", "")
                validation_df.at[idx, "secondary_subcategory"] = secondary.get("subcategory", "")
                validation_df.at[idx, "secondary_subsubcategory"] = secondary.get("subsubcategory", "")
                validation_df.at[idx, "secondary_subsubsubcategory"] = secondary.get("subsubsubcategory", "")
                validation_df.at[idx, "secondary_confidence"] = secondary.get("confidence", "")
                validation_df.at[idx, "secondary_path_valid"] = secondary.get("is_valid_path", False)

            # Tags
            all_tags = result.get("all_tags", [])
            technique_tags = []
            research_tags = []
            functional_tags = []

            for classification in classifications:
                technique_tags.extend(classification.get("technique_tags", []))
                research_tags.extend(classification.get("research_tags", []))
                functional_tags.extend(classification.get("functional_tags", []))

            # Remove duplicates
            technique_tags = list(set(technique_tags))
            research_tags = list(set(research_tags))
            functional_tags = list(set(functional_tags))

            validation_df.at[idx, "technique_tags"] = "|".join(technique_tags)
            validation_df.at[idx, "research_tags"] = "|".join(research_tags)
            validation_df.at[idx, "functional_tags"] = "|".join(functional_tags)
            validation_df.at[idx, "all_tags"] = "|".join(all_tags)
            validation_df.at[idx, "tag_count"] = len(all_tags)

            # Metadata
            validation_df.at[idx, "classification_count"] = result.get("classification_count", 0)
            validation_df.at[idx, "total_token_usage"] = result.get("total_token_usage", 0)
            validation_df.at[idx, "stage1_reasoning"] = result.get("stage1_reasoning", "")

            # Flag for manual review if needed
            needs_review = (
                (classifications and classifications[0].get("confidence", "") == "Low") or
                not any(c.get("is_valid_path", False) for c in classifications) or
                len(all_tags) == 0
            )
            validation_df.at[idx, "needs_manual_review"] = needs_review

    # Save validation results
    validation_df.to_csv(VALIDATION_CSV, index=False)
    logger.info(f"Improved validation sample saved to {VALIDATION_CSV}")

    # Generate report
    generate_validation_report_improved(validation_df)

    return validation_df


def create_validation_sample_improved():
    """Create validation sample with improved structure"""
    try:
        df = pd.read_csv(INPUT_CSV)

        if len(df) < VALIDATION_SAMPLE_SIZE:
            logger.warning(f"Dataset has only {len(df)} products, using all for validation")
            sample_df = df.copy()
        else:
            sample_indices = random.sample(range(len(df)), VALIDATION_SAMPLE_SIZE)
            sample_df = df.iloc[sample_indices].copy()

        # Enhanced classification columns
        sample_df["primary_domain"] = ""
        sample_df["primary_category"] = ""
        sample_df["primary_subcategory"] = ""
        sample_df["primary_subsubcategory"] = ""
        sample_df["primary_subsubsubcategory"] = ""
        sample_df["primary_confidence"] = ""
        sample_df["primary_path_valid"] = False

        sample_df["secondary_domain"] = ""
        sample_df["secondary_category"] = ""
        sample_df["secondary_subcategory"] = ""
        sample_df["secondary_subsubcategory"] = ""
        sample_df["secondary_subsubsubcategory"] = ""
        sample_df["secondary_confidence"] = ""
        sample_df["secondary_path_valid"] = False

        # Tag columns
        sample_df["technique_tags"] = ""
        sample_df["research_tags"] = ""
        sample_df["functional_tags"] = ""
        sample_df["all_tags"] = ""
        sample_df["tag_count"] = 0

        # Metadata
        sample_df["classification_count"] = 0
        sample_df["total_token_usage"] = 0
        sample_df["stage1_reasoning"] = ""
        sample_df["needs_manual_review"] = False

        logger.info(f"Created improved validation sample with {len(sample_df)} products")
        return sample_df

    except Exception as e:
        logger.error(f"Error creating validation sample: {e}")
        raise


def generate_validation_report_improved(validation_df: pd.DataFrame):
    """Generate comprehensive validation report for improved system"""
    print("\n" + "=" * 80)
    print("IMPROVED CLASSIFICATION SYSTEM VALIDATION REPORT")
    print("=" * 80)

    total_products = len(validation_df)
    classified_products = len(
        validation_df[
            validation_df["primary_domain"].notna() & (validation_df["primary_domain"] != "")
        ]
    )

    print(f"Total products validated: {total_products}")
    print(f"Successfully classified: {classified_products} ({classified_products/total_products*100:.1f}%)")

    # Domain distribution
    print(f"\n{'PRIMARY DOMAIN DISTRIBUTION':-^60}")
    domain_counts = validation_df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(10).items():
        print(f"  {domain:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Confidence distribution
    print(f"\n{'CONFIDENCE DISTRIBUTION':-^60}")
    confidence_counts = validation_df["primary_confidence"].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf:<35} {count:>5} ({count/total_products*100:>5.1f}%)")

    # Path validity
    valid_paths = len(validation_df[validation_df["primary_path_valid"] == True])
    print(f"\n{'PATH VALIDITY':-^60}")
    print(f"  Valid classification paths: {valid_paths} ({valid_paths/total_products*100:.1f}%)")

    # Unclassified rate
    unclassified = len(validation_df[validation_df["primary_category"] == "Unclassified"])
    print(f"  Unclassified products: {unclassified} ({unclassified/total_products*100:.1f}%)")

    # Multi-label statistics
    multi_label_count = len(
        validation_df[
            validation_df["secondary_domain"].notna() & (validation_df["secondary_domain"] != "")
        ]
    )
    print(f"\n{'MULTI-LABEL STATISTICS':-^60}")
    print(f"  Products with 2 classifications: {multi_label_count} ({multi_label_count/total_products*100:.1f}%)")
    print(f"  Average tags per product: {validation_df['tag_count'].mean():.1f}")

    # Token usage
    total_tokens = validation_df["total_token_usage"].sum()
    avg_tokens = validation_df["total_token_usage"].mean()
    print(f"\n{'TOKEN USAGE ANALYSIS':-^60}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")


def test_improved_classification():
    """Test the improved classification system with problem cases"""
    
    logging.basicConfig(level=logging.INFO, 
                       format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Initialize systems
    category_system = YAMLCategorySystem()
    tag_system = EnhancedTagSystem()
    classifier = ImprovedTwoStageClassifier(category_system, tag_system)
    
    # Test cases that were failing before
    test_cases = [
        {
            "name": "kyse 150",
            "description": "Human esophageal squamous cell carcinoma cell line",
            "expected_domain": "Cell Biology",
            "expected_path": "Cell Biology -> Primary Cells, Cell Lines and Microorganisms -> Cell Lines"
        },
        {
            "name": "affinityscript multiple temperature cdna synthesis kit",
            "description": "Kit for cDNA synthesis at multiple temperatures",
            "expected_domain": "Cloning and Expression", 
            "expected_path": "Cloning and Expression -> Library Construction -> cDNA Synthesis"
        },
        {
            "name": "rneasy mini rna isolation kit",
            "description": "Kit for RNA isolation from small samples",
            "expected_domain": "Nucleic Acid Purification",
            "expected_path": "Nucleic Acid Purification -> RNA Isolation Kits"
        },
        {
            "name": "anti-bcl-2 antibody",
            "description": "Monoclonal antibody against Bcl-2 protein",
            "expected_domain": "Antibodies",
            "expected_path": "Antibodies -> Primary Antibodies"
        },
        {
            "name": "hela cells",
            "description": "Human cervical cancer cell line",
            "expected_domain": "Cell Biology",
            "expected_path": "Cell Biology -> Primary Cells, Cell Lines and Microorganisms -> Cell Lines"
        }
    ]
    
    print("=" * 80)
    print("TESTING IMPROVED CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTEST {i}: {test_case['name']}")
        print("-" * 50)
        
        result = classifier.classify_product(test_case["name"], test_case["description"])
        
        # Analyze results
        primary_classification = result["classifications"][0] if result["classifications"] else None
        
        if primary_classification:
            actual_domain = primary_classification["domain"]
            actual_path = primary_classification["validated_path"]
            
            print(f"Expected Domain: {test_case['expected_domain']}")
            print(f"Actual Domain:   {actual_domain}")
            domain_match = actual_domain == test_case['expected_domain']
            print(f"Domain Match:    {'✓' if domain_match else '✗'}")
            
            print(f"\nActual Path: {actual_path}")
            print(f"Path Valid:  {'✓' if primary_classification['is_valid_path'] else '✗'}")
            
            print(f"\nTags:")
            print(f"  Technique: {primary_classification.get('technique_tags', [])}")
            print(f"  Research:  {primary_classification.get('research_tags', [])}")
            print(f"  Functional: {primary_classification.get('functional_tags', [])}")
            
            print(f"Confidence: {primary_classification['confidence']}")
            
            if domain_match and primary_classification['is_valid_path']:
                success_count += 1
                print("✅ SUCCESS")
            else:
                print("❌ FAILED")
        else:
            print("❌ NO CLASSIFICATION GENERATED")
        
        print(f"Token Usage: {result.get('total_token_usage', 0)}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS: {success_count}/{len(test_cases)} tests passed ({success_count/len(test_cases)*100:.1f}%)")
    print(f"{'='*80}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("IMPROVED CLASSIFICATION SYSTEM")
    print("=" * 80)

    print(f"Looking for YAML files in: {YAML_DIRECTORY}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Test with problem cases first
        print("\n1. Testing with problem cases...")
        test_improved_classification()

        # Ask user to proceed with validation
        print("\n" + "=" * 80)
        user_input = input("Proceed with full validation sample processing? (y/n): ")

        if user_input.lower() == "y":
            validation_df = process_validation_sample_improved()
            print("\n" + "=" * 80)
            print("VALIDATION COMPLETE!")
            print("=" * 80)

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")
        print("\nTo fix this issue:")
        print("1. Make sure your YAML files are in the correct directory")
        print("2. Check that the directory paths in the configuration are correct")
        print("3. Verify your OpenAI API key is properly configured")


if __name__ == "__main__":
    main()