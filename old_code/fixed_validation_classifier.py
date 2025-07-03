# ENHANCED MULTI-DOMAIN CLASSIFICATION SYSTEM
# This version examines multiple domains and compares their fit before classification

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

# Configuration - update these paths to match your setup
INPUT_CSV = "C:/LabGit/150citations classification/top_50_000_products_in_citations.xlsx - Sheet1.csv"
OUTPUT_CSV = "products_enhanced_classification.csv"
VALIDATION_CSV = "validation_enhanced_classification.csv"
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


class EnhancedCategorySystem:
    """Enhanced category system that builds comprehensive keyword index from actual YAML content"""

    def __init__(self, master_file: str = None, yaml_directory: str = None):
        self.master_file = master_file or MASTER_CATEGORIES_FILE
        self.yaml_directory = yaml_directory or YAML_DIRECTORY
        self.master_config = {}
        self.category_files = {}
        self.domain_keywords = defaultdict(set)
        self.domain_subcategories = defaultdict(set)
        self.domain_structures = {}

        self.load_master_categories()
        self.load_individual_yaml_files()
        self.build_comprehensive_keyword_index()

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

                    # Extract structure and store
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

        # Handle different YAML structures
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

            # Store results
            self.domain_keywords[domain_key] = keywords
            self.domain_subcategories[domain_key] = subcategories

            logger.info(
                f"Built index for {domain_key}: {len(keywords)} keywords, {len(subcategories)} subcategories"
            )

    def _extract_keywords_from_structure(
        self, node, keywords: set, subcategories: set, level=0
    ):
        """Recursively extract keywords from YAML structure"""
        if level > 5:  # Prevent infinite recursion
            return

        if isinstance(node, dict):
            for key, value in node.items():
                # Add the key itself as keywords and subcategory
                key_tokens = self._tokenize_text(key)
                keywords.update(key_tokens)
                subcategories.add(key)

                # Recurse into value
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

        # Clean and normalize
        text = text.lower()
        text = re.sub(r"[^\w\s-]", " ", text)

        # Split on various delimiters
        tokens = set()
        words = re.split(r"[\s\-_/]+", text)

        for word in words:
            word = word.strip()
            if len(word) >= 3:  # Only keep words of 3+ characters
                tokens.add(word)

        return tokens

    def get_candidate_domains_enhanced(
        self, product_name: str, description: str
    ) -> List[Tuple[str, float, str]]:
        """Get candidate domains with enhanced scoring and reasoning"""
        text = f"{product_name} {description}".lower()
        product_tokens = self._tokenize_text(text)

        domain_scores = []

        for domain_key in self.domain_keywords.keys():
            domain_keywords = self.domain_keywords[domain_key]
            domain_subcategories = self.domain_subcategories[domain_key]

            # Calculate keyword overlap score
            keyword_overlap = product_tokens.intersection(domain_keywords)
            subcategory_overlap = set()

            # Check for subcategory matches
            for subcat in domain_subcategories:
                subcat_tokens = self._tokenize_text(subcat)
                if subcat_tokens.intersection(product_tokens):
                    subcategory_overlap.add(subcat)

            # Calculate comprehensive score
            score = 0
            reasoning_parts = []

            # Keyword matching score
            if keyword_overlap:
                keyword_score = len(keyword_overlap) * 2
                score += keyword_score
                reasoning_parts.append(f"keyword matches: {list(keyword_overlap)[:3]}")

            # Subcategory matching score (higher weight)
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

        # Sort by score and return top candidates
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return domain_scores[:6]  # Return top 6 candidates

    def _check_domain_specific_patterns(
        self, product_name: str, description: str, domain_key: str
    ) -> Tuple[float, str]:
        """Check for domain-specific patterns"""
        text = f"{product_name} {description}".lower()

        patterns = {
            "Antibodies": [
                (r"\banti[- ]?[a-z0-9]+\b", 15, "anti-X pattern"),
                (r"\b(monoclonal|polyclonal)\b", 10, "antibody type"),
                (r"\b(primary|secondary)\s+antibody\b", 12, "antibody category"),
                (r"\bigg?[gma]?\b", 8, "immunoglobulin"),
                (r"\b(hrp|fitc|pe|apc|alexa)\b", 8, "conjugation"),
            ],
            "Cell_Biology": [
                (r"\b[a-z]+\s*\d+\s*cell\s*line\b", 20, "cell line pattern"),
                (r"\b(hela|mcf|a549|cho|hek)\b", 18, "known cell line"),
                (r"\bcell\s+(culture|viability|tracking)\b", 12, "cell techniques"),
                (r"\b(transfection|transformation)\b", 10, "cell manipulation"),
            ],
            "PCR": [
                (r"\b(q?pcr|thermocycler|thermal\s+cycler)\b", 15, "PCR equipment"),
                (r"\b(master\s+mix|polymerase|taq)\b", 12, "PCR reagents"),
                (r"\b(primer|probe)\b", 10, "PCR components"),
            ],
            "Assay_Kits": [
                (r"\belisa\s+kit\b", 18, "ELISA kit"),
                (r"\b(cytokine|multiplex)\s+assay\b", 15, "specific assay type"),
                (r"\bassay\s+kit\b", 12, "general assay kit"),
            ],
            "Nucleic_Acid_Purification": [
                (
                    r"\b(rna|dna)\s+(isolation|purification|extraction)\b",
                    18,
                    "nucleic acid purification",
                ),
                (r"\b(genomic|plasmid)\s+dna\b", 15, "DNA type"),
                (r"\brneasy\b", 12, "RNA isolation brand"),
            ],
            "Cloning_And_Expression": [
                (r"\bcdna\s+synthesis\b", 18, "cDNA synthesis"),
                (r"\b(cloning|expression)\s+(kit|vector)\b", 15, "cloning/expression"),
                (r"\b(affinityscript|superscript)\b", 12, "cDNA synthesis brand"),
            ],
        }

        domain_patterns = patterns.get(domain_key, [])
        total_score = 0
        matched_reasons = []

        for pattern, score, reason in domain_patterns:
            if re.search(pattern, text):
                total_score += score
                matched_reasons.append(reason)

        reasoning = f"domain patterns: {matched_reasons}" if matched_reasons else ""
        return total_score, reasoning

    def get_domain_structure_summary(self, domain_key: str, max_items: int = 10) -> str:
        """Get a concise summary of domain structure"""
        if domain_key not in self.domain_structures:
            return f"Domain '{domain_key}' not found"

        structure = self.domain_structures[domain_key]
        subcategories = list(self.domain_subcategories[domain_key])[:max_items]

        lines = [f"DOMAIN: {domain_key.replace('_', ' ')}"]
        lines.append(f"Available subcategories: {', '.join(subcategories[:8])}")
        if len(subcategories) > 8:
            lines.append(f"... and {len(subcategories) - 8} more")

        return "\n".join(lines)

    def get_detailed_structure(self, domain_key: str) -> str:
        """Get detailed structure for a specific domain"""
        if domain_key not in self.domain_structures:
            return f"Domain '{domain_key}' not found"

        structure = self.domain_structures[domain_key]

        def format_structure(node, indent=0, max_depth=4):
            if indent >= max_depth:
                return ["  " * indent + "  • [more subcategories...]"]

            result = []
            prefix = "  " * indent

            if isinstance(node, dict):
                for key, value in list(node.items())[:15]:  # Limit items shown
                    result.append(f"{prefix}- {key}")
                    if value and indent < max_depth - 1:
                        result.extend(format_structure(value, indent + 1, max_depth))
                if len(node) > 15:
                    result.append(f"{prefix}... and {len(node) - 15} more categories")

            elif isinstance(node, list):
                for i, item in enumerate(node[:10]):
                    if isinstance(item, str):
                        result.append(f"{prefix}  • {item}")
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            result.append(f"{prefix}  • {sub_key}")
                            if sub_value and indent < max_depth - 1:
                                result.extend(
                                    format_structure(sub_value, indent + 2, max_depth)
                                )
                            break
                if len(node) > 10:
                    result.append(f"{prefix}  • ... ({len(node)-10} more items)")

            return result

        display_name = domain_key.replace("_", " ")
        lines = [f"STRUCTURE FOR: {display_name.upper()}"]
        lines.append("=" * 50)
        lines.extend(format_structure(structure))
        return "\n".join(lines)

    def validate_classification_path(
        self, domain_key: str, path_components: List[str]
    ) -> Tuple[bool, str]:
        """Validate classification path within domain structure"""
        if domain_key not in self.domain_structures:
            return False, f"Domain '{domain_key}' not found"

        structure = self.domain_structures[domain_key]
        validated_path = [domain_key.replace("_", " ")]

        # Navigate into the domain structure
        current_node = structure

        # Handle top-level domain structure
        if isinstance(current_node, dict) and len(current_node) == 1:
            domain_name = list(current_node.keys())[0]
            current_node = current_node[domain_name]

        logger.info(
            f"Validating path components: {path_components} in domain {domain_key}"
        )

        # Navigate through the structure
        for i, component in enumerate(path_components):
            if not component:
                continue

            found = False

            # Check if we need to navigate into subcategories
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
                    # Fuzzy matching
                    matches = [
                        k for k in current_node.keys() if component.lower() in k.lower()
                    ]
                    if matches:
                        best_match = matches[0]
                        validated_path.append(best_match)
                        current_node = current_node[best_match]
                        found = True
                        logger.info(f"Fuzzy matched '{component}' to '{best_match}'")

                # Navigate deeper if needed
                if found and isinstance(current_node, dict):
                    if "subsubcategories" in current_node:
                        current_node = current_node["subsubcategories"]
                        logger.info(f"Navigated into subsubcategories level")

            elif isinstance(current_node, list):
                # Find in list
                matches = [
                    item
                    for item in current_node
                    if isinstance(item, str) and component.lower() in item.lower()
                ]
                if matches:
                    validated_path.append(matches[0])
                    found = True
                    logger.info(f"Found in list: '{matches[0]}'")

            if not found:
                return (
                    False,
                    f"Invalid path: {' -> '.join(validated_path)} -> '{component}'",
                )

        final_path = " -> ".join(validated_path)
        logger.info(f"Successfully validated path: {final_path}")
        return True, final_path


class EnhancedMultiDomainClassifier:
    """Enhanced classifier that examines multiple domains before deciding"""

    def __init__(self, category_system: EnhancedCategorySystem):
        self.category_system = category_system
        self.logger = logging.getLogger(__name__)

    def classify_product(
        self, product_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Multi-domain classification with comprehensive analysis"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CLASSIFYING: '{product_name}'")
        self.logger.info(f"{'='*60}")

        # Step 1: Get candidate domains with scoring
        candidates = self.category_system.get_candidate_domains_enhanced(
            product_name, description
        )

        self.logger.info(f"Candidate domains found: {len(candidates)}")
        for domain, score, reasoning in candidates[:3]:
            self.logger.info(f"  {domain}: {score} points ({reasoning})")

        if not candidates:
            return self._create_fallback_result(product_name)

        # Step 2: Examine top candidate domains in detail
        domain_analyses = []
        total_tokens = 0

        for domain_key, score, reasoning in candidates[:3]:  # Examine top 3
            analysis = self._analyze_domain_fit(product_name, description, domain_key)
            analysis["candidate_score"] = score
            analysis["candidate_reasoning"] = reasoning
            domain_analyses.append(analysis)
            total_tokens += analysis.get("token_usage", 0)

        # Step 3: Select best domain and classification
        best_analysis = self._select_best_domain_analysis(domain_analyses)

        if not best_analysis:
            return self._create_fallback_result(product_name)

        # Step 4: Format final result
        return {
            "primary_classification": best_analysis,
            "domain_analyses": domain_analyses,
            "candidate_domains": [(d, s) for d, s, r in candidates],
            "total_token_usage": total_tokens,
            "selection_reasoning": best_analysis.get("selection_reasoning", ""),
        }

    def _analyze_domain_fit(
        self, product_name: str, description: str, domain_key: str
    ) -> Dict[str, Any]:
        """Analyze how well a product fits within a specific domain"""

        self.logger.info(f"Analyzing fit for domain: {domain_key}")

        # Get domain structure
        domain_structure = self.category_system.get_detailed_structure(domain_key)

        # Build analysis prompt
        prompt = f"""You are a life science product classification expert.

TASK: Analyze if this product belongs in the {domain_key.replace('_', ' ')} domain and find the best classification path.

{domain_structure}

Product: "{product_name}"
Description: "{description}"

ANALYSIS REQUIREMENTS:
1. Domain Fit Score (0-100): How well does this product fit in this domain?
2. Best Classification Path: Navigate as deep as possible in the structure
3. Confidence Level: High/Medium/Low
4. Alternative Domains: If this product doesn't fit well, suggest 2 better domains

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
    "classification_notes": "any issues or uncertainties"
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

            # Validate the classification path
            path_components = [
                result.get("subcategory", ""),
                result.get("subsubcategory", ""),
                result.get("subsubsubcategory", ""),
            ]
            path_components = [comp for comp in path_components if comp]

            is_valid, validated_path = (
                self.category_system.validate_classification_path(
                    domain_key, path_components
                )
            )

            # Enhance result with validation info
            result.update(
                {
                    "domain": domain_key,
                    "is_valid_path": is_valid,
                    "validated_path": validated_path,
                    "token_usage": (
                        response.usage.total_tokens if hasattr(response, "usage") else 0
                    ),
                }
            )

            self.logger.info(
                f"Domain analysis complete: fit_score={result.get('domain_fit_score', 0)}, valid_path={is_valid}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing domain {domain_key}: {e}")
            return {
                "domain": domain_key,
                "domain_fit_score": 0,
                "belongs_in_domain": False,
                "confidence": "Low",
                "reasoning": f"Analysis failed: {e}",
                "is_valid_path": False,
                "token_usage": 0,
            }

    def _select_best_domain_analysis(self, analyses: List[Dict]) -> Optional[Dict]:
        """Select the best domain analysis based on multiple criteria"""

        if not analyses:
            return None

        # Filter to only domains where the product belongs
        valid_analyses = [a for a in analyses if a.get("belongs_in_domain", False)]

        if not valid_analyses:
            # Fall back to highest scoring analysis
            analyses.sort(key=lambda x: x.get("domain_fit_score", 0), reverse=True)
            best = analyses[0] if analyses else None
        else:
            # Score valid analyses
            for analysis in valid_analyses:
                score = 0

                # Domain fit score (0-100)
                score += analysis.get("domain_fit_score", 0)

                # Path validity bonus
                if analysis.get("is_valid_path", False):
                    score += 20

                # Confidence bonus
                confidence = analysis.get("confidence", "Low")
                if confidence == "High":
                    score += 15
                elif confidence == "Medium":
                    score += 5

                # Candidate score bonus (from initial keyword matching)
                score += analysis.get("candidate_score", 0) * 0.5

                analysis["final_score"] = score

            # Select highest scoring analysis
            valid_analyses.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            best = valid_analyses[0]

        if best:
            # Add selection reasoning
            best["selection_reasoning"] = (
                f"Selected {best.get('domain', '')} (fit: {best.get('domain_fit_score', 0)}, confidence: {best.get('confidence', 'Low')}, valid_path: {best.get('is_valid_path', False)})"
            )

        return best

    def _create_fallback_result(self, product_name: str) -> Dict[str, Any]:
        """Create fallback result when classification fails"""
        return {
            "primary_classification": {
                "domain": "Other",
                "subcategory": "Unclassified",
                "confidence": "Low",
                "reasoning": "No suitable domain found",
                "is_valid_path": False,
                "validated_path": "Other -> Unclassified",
            },
            "domain_analyses": [],
            "candidate_domains": [],
            "total_token_usage": 0,
            "selection_reasoning": "Fallback classification used",
        }

    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def test_enhanced_classification():
    """Test the enhanced classification system"""

    # Initialize systems
    category_system = EnhancedCategorySystem()
    classifier = EnhancedMultiDomainClassifier(category_system)

    # Test cases
    test_cases = [
        {
            "name": "kyse 150",
            "description": "Human esophageal squamous cell carcinoma cell line",
            "expected_domain": "Cell_Biology",
            "expected_contains": "Cell Lines",
        },
        {
            "name": "affinityscript multiple temperature cdna synthesis kit",
            "description": "Kit for cDNA synthesis at multiple temperatures",
            "expected_domain": "Cloning_And_Expression",
            "expected_contains": "cDNA Synthesis",
        },
        {
            "name": "rneasy mini rna isolation kit",
            "description": "Kit for RNA isolation from small samples",
            "expected_domain": "Nucleic_Acid_Purification",
            "expected_contains": "RNA",
        },
        {
            "name": "anti-bcl-2 antibody",
            "description": "Monoclonal antibody against Bcl-2 protein",
            "expected_domain": "Antibodies",
            "expected_contains": "Antibodies",
        },
        {
            "name": "human tnf-alpha elisa kit",
            "description": "ELISA kit for quantifying human TNF-alpha",
            "expected_domain": "Assay_Kits",
            "expected_contains": "ELISA",
        },
    ]

    print("=" * 80)
    print("TESTING ENHANCED MULTI-DOMAIN CLASSIFICATION SYSTEM")
    print("=" * 80)

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTEST {i}: {test_case['name']}")
        print("-" * 50)

        result = classifier.classify_product(
            test_case["name"], test_case["description"]
        )

        # Analyze results
        primary = result.get("primary_classification", {})

        if primary:
            actual_domain = primary.get("domain", "").replace(" ", "_")
            actual_path = primary.get("validated_path", "")

            print(f"Expected Domain: {test_case['expected_domain']}")
            print(f"Actual Domain:   {actual_domain}")
            domain_match = actual_domain == test_case["expected_domain"]
            print(f"Domain Match:    {'✓' if domain_match else '✗'}")

            print(f"\nActual Path: {actual_path}")
            contains_expected = test_case["expected_contains"] in actual_path
            print(f"Contains Expected: {'✓' if contains_expected else '✗'}")

            print(f"Fit Score: {primary.get('domain_fit_score', 0)}")
            print(f"Confidence: {primary.get('confidence', 'Low')}")
            print(f"Path Valid: {'✓' if primary.get('is_valid_path', False) else '✗'}")

            # Show candidate domains
            candidates = result.get("candidate_domains", [])
            print(f"\nCandidate Domains: {[d for d, s in candidates[:3]]}")

            if domain_match and (
                primary.get("is_valid_path", False) or contains_expected
            ):
                success_count += 1
                print("✅ SUCCESS")
            else:
                print("❌ FAILED")
        else:
            print("❌ NO CLASSIFICATION GENERATED")

        print(f"Token Usage: {result.get('total_token_usage', 0)}")

    print(f"\n{'='*80}")
    print(
        f"RESULTS: {success_count}/{len(test_cases)} tests passed ({success_count/len(test_cases)*100:.1f}%)"
    )
    print(f"{'='*80}")

    return success_count >= 4


def process_validation_sample():
    """Process validation sample with enhanced system"""
    logger.info("Starting enhanced validation sample processing...")

    # Initialize systems
    category_system = EnhancedCategorySystem()
    classifier = EnhancedMultiDomainClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        sample_size = min(50, len(df))  # Start with smaller sample
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
        validation_df["validated_path"] = ""
        validation_df["candidate_domains"] = ""
        validation_df["domain_analyses_count"] = 0
        validation_df["total_token_usage"] = 0
        validation_df["selection_reasoning"] = ""
        validation_df["classification_reasoning"] = ""

        logger.info(f"Processing {len(validation_df)} products...")

        for idx in tqdm(validation_df.index, desc="Enhanced Classification"):
            name = validation_df.at[idx, "Name"]
            description = (
                validation_df.at[idx, "Description"]
                if "Description" in validation_df.columns
                else ""
            )

            # Perform classification
            result = classifier.classify_product(name, description)

            # Store results
            primary = result.get("primary_classification", {})

            if primary:
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
                validation_df.at[idx, "validated_path"] = primary.get(
                    "validated_path", ""
                )
                validation_df.at[idx, "classification_reasoning"] = primary.get(
                    "reasoning", ""
                )

            # Store metadata
            candidates = result.get("candidate_domains", [])
            validation_df.at[idx, "candidate_domains"] = "|".join(
                [d for d, s in candidates[:3]]
            )
            validation_df.at[idx, "domain_analyses_count"] = len(
                result.get("domain_analyses", [])
            )
            validation_df.at[idx, "total_token_usage"] = result.get(
                "total_token_usage", 0
            )
            validation_df.at[idx, "selection_reasoning"] = result.get(
                "selection_reasoning", ""
            )

        # Save results
        validation_df.to_csv(VALIDATION_CSV, index=False)
        logger.info(f"Enhanced validation sample saved to {VALIDATION_CSV}")

        # Generate report
        generate_enhanced_validation_report(validation_df)

        return validation_df

    except Exception as e:
        logger.error(f"Error in validation processing: {e}")
        raise


def generate_enhanced_validation_report(validation_df: pd.DataFrame):
    """Generate enhanced validation report"""
    print("\n" + "=" * 80)
    print("ENHANCED MULTI-DOMAIN CLASSIFICATION VALIDATION REPORT")
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

    # Fit score analysis
    fit_scores = validation_df["primary_fit_score"].dropna()
    if len(fit_scores) > 0:
        print(f"\n{'FIT SCORE ANALYSIS':-^60}")
        print(f"  Average fit score: {fit_scores.mean():.1f}")
        print(
            f"  High fit scores (≥80): {len(fit_scores[fit_scores >= 80])} ({len(fit_scores[fit_scores >= 80])/len(fit_scores)*100:.1f}%)"
        )
        print(
            f"  Medium fit scores (50-79): {len(fit_scores[(fit_scores >= 50) & (fit_scores < 80)])} ({len(fit_scores[(fit_scores >= 50) & (fit_scores < 80)])/len(fit_scores)*100:.1f}%)"
        )
        print(
            f"  Low fit scores (<50): {len(fit_scores[fit_scores < 50])} ({len(fit_scores[fit_scores < 50])/len(fit_scores)*100:.1f}%)"
        )

    # Path validity
    valid_paths = len(validation_df[validation_df["primary_path_valid"] == True])
    print(f"\n{'PATH VALIDITY':-^60}")
    print(
        f"  Valid classification paths: {valid_paths} ({valid_paths/total_products*100:.1f}%)"
    )

    # Multi-domain analysis
    analyses_count = validation_df["domain_analyses_count"].dropna()
    if len(analyses_count) > 0:
        print(f"\n{'MULTI-DOMAIN ANALYSIS':-^60}")
        print(f"  Average domains analyzed: {analyses_count.mean():.1f}")
        print(
            f"  Products with multiple candidates: {len(analyses_count[analyses_count > 1])} ({len(analyses_count[analyses_count > 1])/len(analyses_count)*100:.1f}%)"
        )

    # Token usage
    total_tokens = validation_df["total_token_usage"].sum()
    avg_tokens = validation_df["total_token_usage"].mean()
    print(f"\n{'TOKEN USAGE ANALYSIS':-^60}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")

    # Quality metrics
    high_quality = len(
        validation_df[
            (validation_df["primary_confidence"] == "High")
            & (validation_df["primary_path_valid"] == True)
            & (validation_df["primary_fit_score"] >= 70)
        ]
    )

    print(f"\n{'QUALITY METRICS':-^60}")
    print(
        f"  High quality classifications: {high_quality} ({high_quality/total_products*100:.1f}%)"
    )
    print(f"  (High confidence + Valid path + Fit score ≥70)")

    # Show sample results
    print(f"\n{'SAMPLE RESULTS':-^60}")
    sample_results = validation_df[validation_df["primary_path_valid"] == True].head(5)
    for idx, row in sample_results.iterrows():
        print(f"  {row['Name'][:40]:<40} → {row['validated_path']}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("ENHANCED MULTI-DOMAIN CLASSIFICATION SYSTEM")
    print("=" * 80)

    print(f"Looking for master categories file: {MASTER_CATEGORIES_FILE}")
    print(f"Looking for YAML files in: {YAML_DIRECTORY}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Test with enhanced system
        print("\n1. Testing enhanced multi-domain classification...")
        test_success = test_enhanced_classification()

        if test_success:
            print("\n" + "=" * 80)
            user_input = input(
                "Tests passed! Proceed with validation sample processing? (y/n): "
            )

            if user_input.lower() == "y":
                validation_df = process_validation_sample()
                print("\n" + "=" * 80)
                print("🎉 ENHANCED VALIDATION COMPLETE! 🎉")
                print("=" * 80)

                # Ask about full processing
                print(
                    f"\nValidation successful! The enhanced system shows significant improvements:"
                )
                print(f"- Multi-domain candidate analysis")
                print(f"- Comprehensive keyword indexing from actual YAML content")
                print(f"- Domain-specific pattern matching")
                print(f"- Fit scoring and validation")

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
        print("\nTo fix this issue:")
        print("1. Make sure your master_categories.yaml file exists")
        print("2. Make sure your YAML files are in the correct directory")
        print("3. Check that the directory paths in the configuration are correct")
        print("4. Verify your OpenAI API key is properly configured")


def process_full_dataset():
    """Process the full dataset with enhanced classification"""
    logger.info("Starting full dataset processing...")

    # Initialize systems
    category_system = EnhancedCategorySystem()
    classifier = EnhancedMultiDomainClassifier(category_system)

    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df)} products from {INPUT_CSV}")

        # Add classification columns
        df["primary_domain"] = ""
        df["primary_subcategory"] = ""
        df["primary_subsubcategory"] = ""
        df["primary_subsubsubcategory"] = ""
        df["primary_confidence"] = ""
        df["primary_fit_score"] = 0
        df["primary_path_valid"] = False
        df["validated_path"] = ""
        df["candidate_domains"] = ""
        df["total_token_usage"] = 0
        df["classification_reasoning"] = ""

        # Process in batches to manage memory and API usage
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))

            logger.info(
                f"Processing batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx})"
            )

            batch_df = df.iloc[start_idx:end_idx]

            for idx in tqdm(batch_df.index, desc=f"Batch {batch_num + 1}"):
                name = df.at[idx, "Name"]
                description = (
                    df.at[idx, "Description"] if "Description" in df.columns else ""
                )

                try:
                    result = classifier.classify_product(name, description)

                    # Store results
                    primary = result.get("primary_classification", {})

                    if primary:
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
                        df.at[idx, "validated_path"] = primary.get("validated_path", "")
                        df.at[idx, "classification_reasoning"] = primary.get(
                            "reasoning", ""
                        )

                    # Store metadata
                    candidates = result.get("candidate_domains", [])
                    df.at[idx, "candidate_domains"] = "|".join(
                        [d for d, s in candidates[:3]]
                    )
                    df.at[idx, "total_token_usage"] = result.get("total_token_usage", 0)

                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")
                    df.at[idx, "primary_domain"] = "Error"
                    df.at[idx, "classification_reasoning"] = str(e)

            # Save progress after each batch
            df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Saved progress after batch {batch_num + 1}")

        logger.info(f"Full dataset processing complete. Results saved to {OUTPUT_CSV}")

        # Generate final report
        generate_final_report(df)

    except Exception as e:
        logger.error(f"Error in full dataset processing: {e}")
        raise


def generate_final_report(df: pd.DataFrame):
    """Generate final processing report"""
    print("\n" + "=" * 80)
    print("FINAL ENHANCED CLASSIFICATION REPORT")
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

    # Domain distribution
    print(f"\n{'DOMAIN DISTRIBUTION':-^60}")
    domain_counts = df["primary_domain"].value_counts()
    for domain, count in domain_counts.head(15).items():
        print(f"  {domain:<35} {count:>7,} ({count/total_products*100:>5.1f}%)")

    # Quality metrics
    high_quality = len(
        df[
            (df["primary_confidence"] == "High")
            & (df["primary_path_valid"] == True)
            & (df["primary_fit_score"] >= 70)
        ]
    )

    medium_quality = len(
        df[
            (df["primary_confidence"].isin(["High", "Medium"]))
            & (df["primary_fit_score"] >= 50)
        ]
    )

    print(f"\n{'QUALITY ANALYSIS':-^60}")
    print(f"  High quality: {high_quality:,} ({high_quality/total_products*100:.1f}%)")
    print(
        f"  Medium+ quality: {medium_quality:,} ({medium_quality/total_products*100:.1f}%)"
    )

    # Token usage
    total_tokens = df["total_token_usage"].sum()
    print(f"\n{'COST ANALYSIS':-^60}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${total_tokens * 0.00015 / 1000:.2f}")

    print(f"\n{'='*80}")
    print("🎉 ENHANCED CLASSIFICATION SYSTEM COMPLETE! 🎉")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
