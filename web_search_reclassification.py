# WEB SEARCH ENHANCED RE-CLASSIFICATION SYSTEM
# Re-classifies "Other" products using OpenAI web search API

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
from enhanced_classification_dual_w_tagging import (
    LLMDrivenCategorySystem, 
    EnhancedTagSystem,
    EnhancedLLMClassifier
)

# Configuration
INPUT_CSV = "validation_enhanced_fixed_classification_w_tagging_prompt_changes_2.csv"  # Output from enhanced classification
OUTPUT_CSV = "products_web_search_reclassified.csv"
YAML_DIRECTORY = "C:/LabGit/150citations classification/category_structures"
MASTER_CATEGORIES_FILE = "C:/LabGit/150citations classification/master_categories_claude.yaml"

# Web search configuration
MAX_SEARCH_RESULTS = 3
SEARCH_DELAY = 1.0  # Seconds between searches to avoid rate limiting
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
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

class WebSearchProductClassifier:
    """Enhanced classifier that uses web search to gather product information"""
    
    def __init__(self, 
                 category_system: LLMDrivenCategorySystem, 
                 tag_system: EnhancedTagSystem,
                 classifier: EnhancedLLMClassifier):
        self.category_system = category_system
        self.tag_system = tag_system
        self.classifier = classifier
        self.search_cache = {}  # Cache search results to avoid duplicate searches
        self.total_searches = 0
        self.successful_searches = 0
        self.reclassification_stats = defaultdict(int)
        
    def search_product_info(self, product_name: str, manufacturer: str = "") -> Dict[str, Any]:
        """Use OpenAI web search to find product information"""
        
        # Create search query
        search_query = product_name.strip()
        if manufacturer and manufacturer.strip():
            search_query = f"{product_name.strip()} {manufacturer.strip()}"
        
        # Clean up search query
        search_query = re.sub(r'[^\w\s-]', ' ', search_query)  # Remove special chars except hyphens
        search_query = ' '.join(search_query.split())  # Normalize whitespace
        
        # Check cache first
        cache_key = search_query.lower()
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached results for: {search_query}")
            return self.search_cache[cache_key]
        
        logger.info(f"🔍 Searching web for: {search_query}")
        self.total_searches += 1
        
        for attempt in range(MAX_RETRIES):
            try:
                # Use OpenAI's web search capability
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user", 
                            "content": f"""Search the web for detailed information about this life science product: "{search_query}"

Please find and summarize:
1. Product type/category (instrument, reagent, kit, consumable, etc.)
2. Primary application/use (PCR, cell culture, Western blot, etc.)
3. Specific function/purpose
4. Target research areas (cell biology, molecular biology, etc.)
5. Technical specifications if available
6. Any other relevant product details

Focus on factual product information from manufacturer websites, product catalogs, or scientific suppliers."""
                        }
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                search_result = {
                    "query": search_query,
                    "search_successful": True,
                    "product_info": response.choices[0].message.content.strip(),
                    "token_usage": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                    "attempt": attempt + 1
                }
                
                # Cache the result
                self.search_cache[cache_key] = search_result
                self.successful_searches += 1
                
                logger.info(f"✅ Search successful for: {search_query}")
                
                # Rate limiting
                if SEARCH_DELAY > 0:
                    time.sleep(SEARCH_DELAY)
                
                return search_result
                
            except Exception as e:
                logger.warning(f"⚠️ Search attempt {attempt + 1} failed for '{search_query}': {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        # If all attempts failed
        logger.error(f"❌ All search attempts failed for: {search_query}")
        return {
            "query": search_query,
            "search_successful": False,
            "product_info": "",
            "token_usage": 0,
            "error": "Search failed after all retries"
        }
    
    def extract_enhanced_product_info(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured product information from search results"""
        
        if not search_result.get("search_successful", False):
            return {
                "enhanced_description": "",
                "product_type": "Unknown",
                "primary_application": "Unknown",
                "research_areas": [],
                "technical_details": "",
                "confidence": "Low",
                "extraction_successful": False
            }
        
        product_info = search_result.get("product_info", "")
        
        if not product_info.strip():
            return {
                "enhanced_description": "",
                "product_type": "Unknown", 
                "primary_application": "Unknown",
                "research_areas": [],
                "technical_details": "",
                "confidence": "Low",
                "extraction_successful": False
            }
        
        # Use LLM to extract structured information
        extraction_prompt = f"""Extract structured product information from this web search result:

{product_info}

Extract the following information in JSON format:
- product_type: (instrument/equipment, reagent/chemical, kit/assay, consumable/supplies, software, or service)
- primary_application: (main use like PCR, cell culture, Western blot, etc.)
- research_areas: (list of relevant research areas like cell biology, molecular biology, etc.)
- technical_details: (any specific technical information)
- enhanced_description: (comprehensive product description combining all info)
- confidence: (High/Medium/Low based on information quality)

Respond with JSON only:
{{
    "product_type": "instrument",
    "primary_application": "PCR amplification",
    "research_areas": ["molecular biology", "genetics"],
    "technical_details": "real-time PCR with fluorescence detection",
    "enhanced_description": "comprehensive description...",
    "confidence": "High"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=600
            )
            
            text = response.choices[0].message.content.strip()
            cleaned = self._strip_code_fences(text)
            extraction_result = json.loads(cleaned)
            
            extraction_result["extraction_successful"] = True
            extraction_result["extraction_token_usage"] = (
                response.usage.total_tokens if hasattr(response, 'usage') else 0
            )
            
            logger.info(f"✅ Extracted product type: {extraction_result.get('product_type', 'Unknown')}")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"❌ Failed to extract product info: {e}")
            return {
                "enhanced_description": product_info,  # Use raw search result as fallback
                "product_type": "Unknown",
                "primary_application": "Unknown", 
                "research_areas": [],
                "technical_details": "",
                "confidence": "Low",
                "extraction_successful": False,
                "extraction_token_usage": 0,
                "error": str(e)
            }
    
    def reclassify_with_web_info(self, 
                                product_name: str, 
                                manufacturer: str,
                                original_description: str = "") -> Dict[str, Any]:
        """Complete reclassification workflow with web search enhancement"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🌐 WEB SEARCH RECLASSIFICATION: '{product_name}'")
        logger.info(f"{'='*60}")
        
        total_tokens = 0
        
        # Step 1: Web Search
        logger.info("🔍 STEP 1: Web Search")
        search_result = self.search_product_info(product_name, manufacturer)
        total_tokens += search_result.get("token_usage", 0)
        
        if not search_result.get("search_successful", False):
            logger.warning(f"⚠️ Web search failed for '{product_name}', using original classification")
            return self._create_failed_search_result(product_name, original_description, total_tokens)
        
        # Step 2: Extract Enhanced Information  
        logger.info("📝 STEP 2: Extract Product Information")
        enhanced_info = self.extract_enhanced_product_info(search_result)
        total_tokens += enhanced_info.get("extraction_token_usage", 0)
        
        if not enhanced_info.get("extraction_successful", False):
            logger.warning(f"⚠️ Information extraction failed for '{product_name}'")
            return self._create_failed_extraction_result(product_name, search_result, total_tokens)
        
        # Step 3: Enhanced Classification
        logger.info("🎯 STEP 3: Enhanced Classification")
        
        # Combine original and enhanced descriptions
        combined_description = self._create_combined_description(
            original_description, 
            enhanced_info.get("enhanced_description", "")
        )
        
        # Use the existing classifier with enhanced information
        classification_result = self.classifier.classify_product(
            product_name, 
            combined_description
        )
        
        total_tokens += classification_result.get("total_token_usage", 0)
        
        # Add web search metadata
        classification_result.update({
            "web_search_performed": True,
            "web_search_successful": True,
            "original_domain": "Other",
            "search_query": search_result.get("query", ""),
            "enhanced_product_info": enhanced_info,
            "web_search_tokens": search_result.get("token_usage", 0),
            "extraction_tokens": enhanced_info.get("extraction_token_usage", 0),
            "total_web_enhanced_tokens": total_tokens,
            "reclassification_confidence": enhanced_info.get("confidence", "Low")
        })
        
        # Update stats
        new_domain = classification_result.get("primary_classification", {}).get("domain", "Other")
        self.reclassification_stats[f"Other -> {new_domain}"] += 1
        
        logger.info(f"✅ Reclassified from 'Other' to '{new_domain}'")
        
        return classification_result
    
    def _create_combined_description(self, original: str, enhanced: str) -> str:
        """Combine original and enhanced descriptions intelligently"""
        parts = []
        
        if original and original.strip():
            parts.append(f"Original: {original.strip()}")
        
        if enhanced and enhanced.strip():
            parts.append(f"Enhanced: {enhanced.strip()}")
        
        return " | ".join(parts) if parts else ""
    
    def _create_failed_search_result(self, product_name: str, description: str, tokens: int) -> Dict[str, Any]:
        """Create result for failed web search"""
        return {
            "web_search_performed": True,
            "web_search_successful": False,
            "original_domain": "Other",
            "primary_classification": {
                "domain": "Other",
                "subcategory": "Unclassified",
                "confidence": "Low",
                "reasoning": "Web search failed, insufficient information",
                "is_primary": True,
                "validated_path": "Other -> Unclassified"
            },
            "classifications": [],
            "total_web_enhanced_tokens": tokens,
            "reclassification_confidence": "Low"
        }
    
    def _create_failed_extraction_result(self, product_name: str, search_result: Dict, tokens: int) -> Dict[str, Any]:
        """Create result for failed information extraction"""
        return {
            "web_search_performed": True,
            "web_search_successful": True,
            "extraction_successful": False,
            "original_domain": "Other",
            "primary_classification": {
                "domain": "Other", 
                "subcategory": "Unclassified",
                "confidence": "Low",
                "reasoning": "Information extraction failed",
                "is_primary": True,
                "validated_path": "Other -> Unclassified"
            },
            "classifications": [],
            "search_query": search_result.get("query", ""),
            "raw_search_info": search_result.get("product_info", ""),
            "total_web_enhanced_tokens": tokens,
            "reclassification_confidence": "Low"
        }
    
    def _strip_code_fences(self, text: str) -> str:
        """Remove code fences from LLM response"""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def load_and_filter_other_products(csv_file: str) -> pd.DataFrame:
    """Load CSV and filter for products classified as 'Other'"""
    logger.info(f"📊 Loading classification results from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"✅ Loaded {len(df)} total products")
        
        # Filter for "Other" classifications
        other_mask = (df['primary_domain'] == 'Other') | (df['primary_domain'].isna())
        other_products = df[other_mask].copy()
        
        logger.info(f"🎯 Found {len(other_products)} products classified as 'Other' ({len(other_products)/len(df)*100:.1f}%)")
        
        if len(other_products) == 0:
            logger.warning("⚠️ No products found with 'Other' classification")
            return pd.DataFrame()
        
        # Show sample of products to be reclassified
        logger.info("📋 Sample products to be reclassified:")
        for idx, row in other_products.head(5).iterrows():
            name = row.get('Name', 'Unknown')[:50]
            manufacturer = row.get('Manufacturer', 'Unknown')[:30]
            logger.info(f"  - {name} | {manufacturer}")
        
        return other_products
        
    except Exception as e:
        logger.error(f"❌ Error loading CSV: {e}")
        raise


def process_web_search_reclassification(input_csv: str, 
                                      output_csv: str,
                                      max_products: int = None,
                                      test_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main processing function for web search reclassification"""
    
    logger.info("🌐 Starting Web Search Enhanced Reclassification...")
    
    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem() 
    base_classifier = EnhancedLLMClassifier(category_system, tag_system)
    web_classifier = WebSearchProductClassifier(category_system, tag_system, base_classifier)
    
    # Load complete dataset AND filter "Other" products
    logger.info(f"📊 Loading complete dataset from: {input_csv}")
    complete_df = pd.read_csv(input_csv)
    logger.info(f"✅ Loaded {len(complete_df)} total products")
    
    other_products_df = load_and_filter_other_products(input_csv)
    
    if len(other_products_df) == 0:
        logger.info("✅ No products need reclassification")
        return other_products_df
    
    # Limit products for testing
    if test_mode and max_products:
        other_products_df = other_products_df.head(max_products)
        logger.info(f"🧪 Test mode: Processing {len(other_products_df)} products")
    
    # Add new columns for web search results
    web_search_columns = [
        "web_search_performed",
        "web_search_successful", 
        "search_query",
        "enhanced_product_type",
        "enhanced_primary_application",
        "enhanced_research_areas", 
        "enhanced_description_full",
        "reclassification_confidence",
        "new_primary_domain",
        "new_primary_subcategory",
        "new_primary_subsubcategory", 
        "new_primary_subsubsubcategory",
        "new_primary_confidence",
        "new_primary_fit_score",
        "new_validated_path",
        "new_technique_tags",
        "new_research_tags",
        "new_functional_tags",
        "new_total_tags",
        "web_search_tokens",
        "extraction_tokens", 
        "reclassification_tokens",
        "total_web_enhanced_tokens"
    ]
    
    for col in web_search_columns:
        if col.endswith('_tokens') or col.endswith('_score'):
            other_products_df[col] = 0
        elif col.endswith('_performed') or col.endswith('_successful'):
            other_products_df[col] = False
        else:
            other_products_df[col] = ""
    
    # Process each product
    logger.info(f"🔄 Processing {len(other_products_df)} products with web search reclassification...")
    
    successful_reclassifications = 0
    failed_searches = 0
    
    for idx in tqdm(other_products_df.index, desc="🌐 Web Search Reclassification"):
        name = other_products_df.at[idx, 'Name']
        manufacturer = other_products_df.at[idx, 'Manufacturer'] if 'Manufacturer' in other_products_df.columns else ""
        original_description = other_products_df.at[idx, 'Description'] if 'Description' in other_products_df.columns else ""
        
        try:
            # Perform web search reclassification
            result = web_classifier.reclassify_with_web_info(name, manufacturer, original_description)
            
            # Store web search metadata
            other_products_df.at[idx, 'web_search_performed'] = result.get('web_search_performed', False)
            other_products_df.at[idx, 'web_search_successful'] = result.get('web_search_successful', False)
            other_products_df.at[idx, 'search_query'] = result.get('search_query', '')
            other_products_df.at[idx, 'reclassification_confidence'] = result.get('reclassification_confidence', 'Low')
            
            # Store enhanced product info
            enhanced_info = result.get('enhanced_product_info', {})
            other_products_df.at[idx, 'enhanced_product_type'] = enhanced_info.get('product_type', '')
            other_products_df.at[idx, 'enhanced_primary_application'] = enhanced_info.get('primary_application', '')
            other_products_df.at[idx, 'enhanced_research_areas'] = '|'.join(enhanced_info.get('research_areas', []))
            other_products_df.at[idx, 'enhanced_description_full'] = enhanced_info.get('enhanced_description', '')
            
            # Store new classification results
            classifications = result.get('classifications', [])
            if classifications:
                primary = classifications[0]
                other_products_df.at[idx, 'new_primary_domain'] = primary.get('domain', '')
                other_products_df.at[idx, 'new_primary_subcategory'] = primary.get('subcategory', '')
                other_products_df.at[idx, 'new_primary_subsubcategory'] = primary.get('subsubcategory', '')
                other_products_df.at[idx, 'new_primary_subsubsubcategory'] = primary.get('subsubsubcategory', '')
                other_products_df.at[idx, 'new_primary_confidence'] = primary.get('confidence', '')
                other_products_df.at[idx, 'new_primary_fit_score'] = primary.get('domain_fit_score', 0)
                other_products_df.at[idx, 'new_validated_path'] = primary.get('validated_path', '')
                
                # Store tags
                other_products_df.at[idx, 'new_technique_tags'] = '|'.join(primary.get('technique_tags', []))
                other_products_df.at[idx, 'new_research_tags'] = '|'.join(primary.get('research_tags', []))
                other_products_df.at[idx, 'new_functional_tags'] = '|'.join(primary.get('functional_tags', []))
                other_products_df.at[idx, 'new_total_tags'] = primary.get('total_tags', 0)
            
            # Store token usage
            other_products_df.at[idx, 'web_search_tokens'] = result.get('web_search_tokens', 0)
            other_products_df.at[idx, 'extraction_tokens'] = result.get('extraction_tokens', 0)
            other_products_df.at[idx, 'reclassification_tokens'] = result.get('total_token_usage', 0)
            other_products_df.at[idx, 'total_web_enhanced_tokens'] = result.get('total_web_enhanced_tokens', 0)
            
            # Track success
            if result.get('web_search_successful', False):
                new_domain = other_products_df.at[idx, 'new_primary_domain']
                if new_domain and new_domain != 'Other':
                    successful_reclassifications += 1
                else:
                    failed_searches += 1
            else:
                failed_searches += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing '{name}': {e}")
            other_products_df.at[idx, 'web_search_performed'] = True
            other_products_df.at[idx, 'web_search_successful'] = False
            other_products_df.at[idx, 'reclassification_confidence'] = 'Error'
            failed_searches += 1
    
    # Save reclassification details (just the "Other" products with web search results)
    reclassification_details_csv = output_csv.replace('.csv', '_reclassification_details.csv')
    other_products_df.to_csv(reclassification_details_csv, index=False)
    logger.info(f"✅ Reclassification details saved to: {reclassification_details_csv}")
    
    # Create complete updated dataset by merging results back
    logger.info("🔄 Merging reclassified results back into complete dataset...")
    complete_updated_df = merge_reclassified_results(complete_df, other_products_df)
    
    # Save complete updated dataset
    complete_updated_csv = output_csv.replace('.csv', '_complete_updated.csv') 
    complete_updated_df.to_csv(complete_updated_csv, index=False)
    logger.info(f"✅ Complete updated dataset saved to: {complete_updated_csv}")
    
    # Generate report
    generate_web_search_reclassification_report(other_products_df, web_classifier)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🌐 WEB SEARCH RECLASSIFICATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Products processed: {len(other_products_df)}")
    logger.info(f"✅ Successful reclassifications: {successful_reclassifications}")
    logger.info(f"❌ Failed searches/classifications: {failed_searches}")
    logger.info(f"📈 Success rate: {successful_reclassifications/len(other_products_df)*100:.1f}%")
    logger.info(f"🔍 Total web searches: {web_classifier.total_searches}")
    logger.info(f"✅ Successful searches: {web_classifier.successful_searches}")
    logger.info(f"\n📁 OUTPUT FILES:")
    logger.info(f"  1. Reclassification details: {reclassification_details_csv}")
    logger.info(f"  2. Complete updated dataset: {complete_updated_csv}")
    
    return complete_updated_df, other_products_df


def merge_reclassified_results(complete_df: pd.DataFrame, reclassified_df: pd.DataFrame) -> pd.DataFrame:
    """Merge reclassified 'Other' products back into the complete dataset"""
    
    logger.info(f"🔄 Merging {len(reclassified_df)} reclassified products into {len(complete_df)} total products")
    
    # Create a copy of the complete dataset
    updated_df = complete_df.copy()
    
    # Add web search columns to complete dataset (initialize with default values)
    web_search_columns = [
        "web_search_performed",
        "web_search_successful", 
        "search_query",
        "enhanced_product_type",
        "enhanced_primary_application",
        "enhanced_research_areas", 
        "enhanced_description_full",
        "reclassification_confidence",
        "new_primary_domain",
        "new_primary_subcategory",
        "new_primary_subsubcategory", 
        "new_primary_subsubsubcategory",
        "new_primary_confidence",
        "new_primary_fit_score",
        "new_validated_path",
        "new_technique_tags",
        "new_research_tags",
        "new_functional_tags",
        "new_total_tags",
        "web_search_tokens",
        "extraction_tokens", 
        "reclassification_tokens",
        "total_web_enhanced_tokens"
    ]
    
    # Initialize web search columns for all products
    for col in web_search_columns:
        if col.endswith('_tokens') or col.endswith('_score'):
            updated_df[col] = 0
        elif col.endswith('_performed') or col.endswith('_successful'):
            updated_df[col] = False
        else:
            updated_df[col] = ""
    
    # Update products that were reclassified
    successful_merges = 0
    
    for idx, reclassified_row in reclassified_df.iterrows():
        # Find matching row in complete dataset (by index is most reliable)
        if idx in updated_df.index:
            # Copy web search results
            for col in web_search_columns:
                if col in reclassified_row:
                    updated_df.at[idx, col] = reclassified_row[col]
            
            # Update primary classification if reclassification was successful
            if (reclassified_row.get('web_search_successful', False) and 
                reclassified_row.get('new_primary_domain', '') not in ['', 'Other']):
                
                # Update primary classification columns with new results
                updated_df.at[idx, 'primary_domain'] = reclassified_row.get('new_primary_domain', '')
                updated_df.at[idx, 'primary_subcategory'] = reclassified_row.get('new_primary_subcategory', '')
                updated_df.at[idx, 'primary_subsubcategory'] = reclassified_row.get('new_primary_subsubcategory', '')
                updated_df.at[idx, 'primary_subsubsubcategory'] = reclassified_row.get('new_primary_subsubsubcategory', '')
                updated_df.at[idx, 'primary_confidence'] = reclassified_row.get('new_primary_confidence', '')
                updated_df.at[idx, 'primary_fit_score'] = reclassified_row.get('new_primary_fit_score', 0)
                updated_df.at[idx, 'validated_path_primary'] = reclassified_row.get('new_validated_path', '')
                
                # Update tags
                updated_df.at[idx, 'technique_tags'] = reclassified_row.get('new_technique_tags', '')
                updated_df.at[idx, 'research_tags'] = reclassified_row.get('new_research_tags', '')
                updated_df.at[idx, 'functional_tags'] = reclassified_row.get('new_functional_tags', '')
                updated_df.at[idx, 'total_tags'] = reclassified_row.get('new_total_tags', 0)
                
                successful_merges += 1
                
                logger.info(f"✅ Updated {updated_df.at[idx, 'Name'][:30]} → {reclassified_row.get('new_primary_domain', '')}")
    
    logger.info(f"🔄 Successfully merged {successful_merges} reclassified products")
    
    return updated_df


def generate_web_search_reclassification_report(df: pd.DataFrame, classifier: WebSearchProductClassifier):
    """Generate comprehensive report on web search reclassification results"""
    
    print(f"\n{'='*80}")
    print("🌐 WEB SEARCH RECLASSIFICATION DETAILED REPORT")
    print(f"{'='*80}")
    
    total_products = len(df)
    successful_searches = len(df[df['web_search_successful'] == True])
    successful_reclassifications = len(df[
        (df['web_search_successful'] == True) & 
        (df['new_primary_domain'] != '') & 
        (df['new_primary_domain'] != 'Other')
    ])
    
    print(f"📊 OVERVIEW")
    print(f"  Total products processed: {total_products}")
    print(f"  Successful web searches: {successful_searches} ({successful_searches/total_products*100:.1f}%)")
    print(f"  Successful reclassifications: {successful_reclassifications} ({successful_reclassifications/total_products*100:.1f}%)")
    
    # New domain distribution
    print(f"\n{'NEW DOMAIN DISTRIBUTION':-^60}")
    new_domain_counts = df[df['new_primary_domain'] != '']['new_primary_domain'].value_counts()
    for domain, count in new_domain_counts.head(10).items():
        print(f"  {domain:<35} {count:>5} ({count/total_products*100:>5.1f}%)")
    
    # Confidence analysis
    print(f"\n{'RECLASSIFICATION CONFIDENCE':-^60}")
    confidence_counts = df['reclassification_confidence'].value_counts()
    for conf, count in confidence_counts.items():
        if conf:
            print(f"  {conf:<20} {count:>5} ({count/total_products*100:>5.1f}%)")
    
    # Enhanced product type distribution
    print(f"\n{'ENHANCED PRODUCT TYPES':-^60}")
    product_type_counts = df[df['enhanced_product_type'] != '']['enhanced_product_type'].value_counts()
    for ptype, count in product_type_counts.head(8).items():
        print(f"  {ptype:<35} {count:>5}")
    
    # Token usage analysis
    print(f"\n{'TOKEN USAGE ANALYSIS':-^60}")
    total_tokens = df['total_web_enhanced_tokens'].sum()
    avg_tokens_per_product = df['total_web_enhanced_tokens'].mean()
    avg_search_tokens = df['web_search_tokens'].mean()
    avg_extraction_tokens = df['extraction_tokens'].mean()
    
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average tokens per product: {avg_tokens_per_product:.1f}")
    print(f"  Average search tokens: {avg_search_tokens:.1f}")
    print(f"  Average extraction tokens: {avg_extraction_tokens:.1f}")
    print(f"  Estimated cost (GPT-4o-mini): ${total_tokens * 0.00015 / 1000:.4f}")
    
    # Show excellent reclassifications
    excellent_reclassifications = df[
        (df['reclassification_confidence'] == 'High') &
        (df['new_primary_domain'] != 'Other') &
        (df['new_primary_domain'] != '')
    ].head(5)
    
    if len(excellent_reclassifications) > 0:
        print(f"\n{'🌟 EXCELLENT RECLASSIFICATIONS':-^60}")
        for idx, row in excellent_reclassifications.iterrows():
            name = row['Name'][:30]
            new_domain = row['new_primary_domain']
            enhanced_type = row['enhanced_product_type']
            print(f"  {name:<30} → {new_domain} ({enhanced_type})")
    
    # Show reclassification patterns
    print(f"\n{'RECLASSIFICATION PATTERNS':-^60}")
    for pattern, count in classifier.reclassification_stats.items():
        print(f"  {pattern:<50} {count:>5}")


def test_web_search_reclassification():
    """Test the web search reclassification system"""
    
    print("=" * 80)
    print("🧪 TESTING WEB SEARCH RECLASSIFICATION SYSTEM")
    print("=" * 80)
    
    # Test with a few sample "Other" products
    test_products = [
        {"name": "BD FACSCanto II", "manufacturer": "BD Biosciences"},
        {"name": "Giemsa stain", "manufacturer": "Sigma-Aldrich"}, 
        {"name": "PVDF membrane", "manufacturer": "Millipore"},
        {"name": "Human TNF-alpha ELISA Kit", "manufacturer": "R&D Systems"},
        {"name": "Acetonitrile HPLC grade", "manufacturer": "Fisher Scientific"}
    ]
    
    # Initialize systems
    category_system = LLMDrivenCategorySystem()
    tag_system = EnhancedTagSystem()
    base_classifier = EnhancedLLMClassifier(category_system, tag_system)
    web_classifier = WebSearchProductClassifier(category_system, tag_system, base_classifier)
    
    success_count = 0
    
    for i, test_product in enumerate(test_products, 1):
        print(f"\n🧪 TEST {i}: {test_product['name']}")
        print("-" * 50)
        
        try:
            result = web_classifier.reclassify_with_web_info(
                test_product['name'], 
                test_product['manufacturer']
            )
            
            success = result.get('web_search_successful', False)
            new_domain = result.get('primary_classification', {}).get('domain', 'Other')
            confidence = result.get('reclassification_confidence', 'Low')
            
            print(f"Search successful: {'✅' if success else '❌'}")
            print(f"New domain: {new_domain}")
            print(f"Confidence: {confidence}")
            
            if success and new_domain != 'Other':
                success_count += 1
                print("✅ RECLASSIFICATION: SUCCESS")
            else:
                print("❌ RECLASSIFICATION: FAILED")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"🧪 TEST RESULTS: {success_count}/{len(test_products)} passed ({success_count/len(test_products)*100:.1f}%)")
    print(f"{'='*80}")
    
    return success_count >= 3


def main():
    """Main execution function"""
    print("=" * 80) 
    print("🌐 WEB SEARCH ENHANCED RE-CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    try:
        print("\n1. Testing web search reclassification system...")
        test_success = test_web_search_reclassification()
        
        if test_success:
            print("\n" + "=" * 80)
            user_input = input("✅ Tests passed! Run reclassification on validation data? (y/n): ")
            
            if user_input.lower() == 'y':
                # Process validation sample
                complete_df, reclassified_df = process_web_search_reclassification(
                    input_csv=INPUT_CSV,
                    output_csv=OUTPUT_CSV,
                    max_products=20,  # Start with small sample
                    test_mode=True
                )
                
                if len(reclassified_df) > 0:
                    full_input = input("\n🚀 Run on all 'Other' products? (y/n): ")
                    if full_input.lower() == 'y':
                        # Process all "Other" products
                        process_web_search_reclassification(
                            input_csv=INPUT_CSV,
                            output_csv=OUTPUT_CSV.replace('.csv', '_full.csv'),
                            test_mode=False
                        )
            else:
                print("Testing complete.")
        else:
            print("\n❌ Tests failed. Check the error messages above.")
                
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()