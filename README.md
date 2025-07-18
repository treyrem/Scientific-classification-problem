Life Science Product Classification System - Comprehensive Guide
This is a sophisticated 3-stage chain prompting classification system designed to automatically categorize life science products into detailed hierarchical taxonomies with cross-cutting tags.  how it works:

🏗️ System Architecture Overview
Core Components

Master Configuration System (master_categories_claude.yaml)
Domain-Specific YAML Structures  (category_structures folder referenced by master config)
3-Stage Classification Pipeline (enhanced_classification_full_checkpoint.py)
Web Search Reclassification (web_search_integration.py)
Checkpointing System (for processing large datasets)


📋 Master Categories File - The Control Center
The master_categories_claude.yaml file is the central nervous system that:
Maps Domains to YAML Files
yamldomain_mapping:
  Cell_Biology:
    yaml_file: "category_structure_cell_biology.yaml"
    description: "Cell culture systems, cell analysis tools..."
    keywords: ["cell", "giemsa", "crystal violet"...]
Provides Domain Intelligence

Keywords: Help identify which domain a product belongs to
Typical Products: Examples of what fits in each domain
Classification Hints: Rules for edge cases
Dual Classification Opportunities: Where products can fit multiple domains

Defines 19 Major Domains

Cell_Biology - Cell culture, staining, flow cytometry
Lab_Equipment - Physical instruments, analyzers, systems
Antibodies - Primary, secondary, research antibodies
Chemistry - Solvents, acids, salts, basic chemicals
Assay_Kits - Complete assay systems (ELISA, cell viability)
Protein - Western blot supplies, protein analysis
Molecular_Biology - DNA/RNA reagents, enzymes
PCR - PCR reagents and consumables
And 9 more specialized domains...


🔄 3-Stage Classification Pipeline
Stage 1: Intelligent Domain Selection
pythondef select_domain_with_llm(self, product_name: str, description: str)
What it does:

Uses GPT-4o-mini to analyze product name + description
Selects 1-2 BEST domains from 19 available domains
Can identify dual-function products (e.g., antibody + imaging)
Returns primary domain + optional secondary domain with confidence scores

Key Intelligence:

Product Type beats Application - Instruments → Lab Equipment, not by what they analyze
Equipment vs Reagent Distinction - Physical instruments vs chemicals/biologicals
Mandatory Consumables Rules - Tubes, plates, membranes → Lab Equipment
Enhanced Kit Classification - Complete assays vs preparation kits

Stage 2a/2b: Deep Hierarchical Classification
pythondef _classify_within_domain(self, product_name: str, description: str, domain_key: str)
What it does:

For each selected domain, classifies to deepest possible level
Uses domain-specific YAML structure for context
Aims for 3-4 level deep classifications: Domain → Subcategory → Subsubcategory → Subsubsubcategory
Validates paths against actual YAML structures

Example Output:
Primary: Antibodies → Research Area Antibodies → Neuroscience Antibodies → Neuronal Marker Antibodies → NeuN Antibodies
Secondary: Cell_Biology → Cell Analysis → Flow Cytometry
Stage 3: Unified Cross-Cutting Tagging
pythondef tag_product(self, product_name: str, description: str, ...)
What it does:

Applies 4 categories of tags considering BOTH domain classifications:

Technique Tags: HOW it's used (Western Blot, Flow Cytometry, PCR)
Research Application Tags: WHAT research it supports (Cancer Research, Neuroscience)
Functional Tags: WHAT TYPE it is (Kit, Instrument, High-Throughput)
Specimen Tags: WHAT species it works with (Human, Mouse, Rat)




🔍 How Domain → YAML File Mapping Works
1. Master Config References YAML Files
yamlCell_Biology:
  yaml_file: "category_structure_cell_biology.yaml"
2. System Loads Individual YAML Files
pythondef load_individual_yaml_files(self):
    for domain_key, domain_info in domain_mapping.items():
        yaml_file = domain_info.get("yaml_file")
        yaml_path = os.path.join(self.yaml_directory, yaml_file)
        # Load and parse YAML structure
3. Extracts Deep Classification Paths
pythondef _extract_actual_paths_from_yaml(self, domain_key: str) -> List[str]:
    # Extracts all possible classification paths like:
    # "Cell-Based Assays → ATP Assay Kits → Luciferase ATP Assays"
4. Uses Paths in Classification Prompts
The system shows the LLM ALL available paths from the YAML for ultra-specific classification:
COMPREHENSIVE CLASSIFICATION PATHS (150+ total paths):

2-LEVEL PATHS:
- Cell-Based Assays → ATP Assay Kits  
- ELISA Kits → Cytokine/Chemokine ELISA Kits

3-LEVEL PATHS:  
- Cell-Based Assays → ATP Assay Kits → Luciferase ATP Assays
- Primary Antibodies → Monoclonal Antibodies → IgG Isotype Controls

🔧 Web Search Reclassification System
When It's Used
The web search system (web_search_integration.py) handles problematic classifications:

Products classified as "Other" domain
Generic classifications like "Analytical Instrumentation"
Misclassified antibiotics (streptomycin as acid instead of antibiotic)
Generic software classifications

How It Works
pythondef correct_web_search_and_identify_domain(self, product_name, manufacturer, problem_type):

Real Web Search: Uses OpenAI's Responses API with web search tools
Enhanced Context: Gets official product specs, datasheets, manufacturer info
Re-classification: Uses web search results to make better domain + path decisions
Fallback: If web search fails, uses LLM knowledge as backup


💾 Checkpointing System
Why Checkpointing?

Processing 50,000+ products takes hours
API failures can lose progress
Allows resuming from interruptions

How It Works
pythondef find_latest_checkpoint():
    # Finds most recent checkpoint file
    # Returns filepath and progress number

# Saves every 50 products:
if (idx + 1) % CHECKPOINT_FREQ == 0:
    checkpoint_file = f"enhanced_validation_checkpoint_{idx + 1}.csv"
    df.to_csv(checkpoint_file, index=False)
Checkpoint Types

Validation Checkpoints: For 100-product test samples
Full Dataset Checkpoints: For complete 50K product processing
Web Search Checkpoints: For problematic product reclassification


📊 Output Structure
Primary Classification Columns
primary_domain, primary_subcategory, primary_subsubcategory, primary_subsubsubcategory
primary_confidence, primary_fit_score, validated_path_primary
Secondary Classification Columns (for dual-function products)
secondary_domain, secondary_subcategory, secondary_subsubcategory, secondary_subsubsubcategory  
secondary_confidence, secondary_fit_score, validated_path_secondary
Tagging Columns (4 tag categories)
technique_tags, research_tags, functional_tags, specimen_tags
all_tags, total_tags, tag_confidence
Metadata Columns
is_dual_function, classification_count, total_token_usage, chain_prompting_stages
error_occurred, domain_selection_primary_confidence

🚀 Processing Workflows
1. Validation Workflow (100 products)
pythondef process_enhanced_validation_sample():
    # Tests system on 100 random products
    # Generates detailed performance reports
    # Shows examples of excellent classifications
2. Full Dataset Workflow (50,000+ products)
pythondef process_full_dataset():
    # Processes complete product catalog
    # Uses checkpointing for reliability
    # Handles errors gracefully
3. Web Search Reclassification Workflow
pythondef process_corrected_real_web_search_reclassification_with_checkpointing():
    # Identifies problematic products from previous runs
    # Uses web search to get better information
    # Re-classifies with enhanced context

🔍 Key Algorithms & Intelligence
Domain Name Mapping
pythondef _initialize_domain_mapping(self):
    # Handles domain name variations
    # Maps "Protein" → "Protein Biochemistry"
Trusted Pattern Recognition
pythondef _is_trusted_llm_path(self, domain_key: str, path_components: List[str]):
    # Recognizes good LLM suggestions
    # Validates against known good patterns
YAML Structure Navigation
pythondef _extract_structure(self, data):
    # Handles different YAML formats
    # Extracts categories/subcategories/items
Path Validation
pythondef validate_classification_path(self, domain_key: str, path_components: List[str]):
    # Validates LLM suggestions against YAML structure
    # Uses fuzzy matching for minor variations
    # Falls back to partial paths when needed

💡 Key Innovations
1. Chain Prompting Architecture
Instead of one massive prompt, uses 3 focused stages:

Stage 1: Domain selection (focused on domain characteristics)
Stage 2: Deep classification (focused on hierarchical structure)
Stage 3: Tagging (focused on cross-cutting attributes)

2. Dual Classification Support
Can identify products that legitimately fit multiple domains:

NeuN antibodies: Antibodies + Cell_Biology
qPCR machines: Lab_Equipment + Molecular_Diagnostics

3. Comprehensive YAML Integration

Shows LLM ALL possible paths for ultra-specific classification
No truncation - complete context for best decisions
Validates against actual structure, not assumed structure

4. Intelligent Fallback Systems

Web search for problematic classifications
LLM knowledge backup when web search fails
Partial path acceptance when full validation fails

5. Production-Ready Reliability

Comprehensive error handling
Checkpointing for large datasets
Token usage tracking
Detailed performance analytics


🎯 Usage for New Team Member
Start Here:

Review master_categories_claude.yaml - Understand the 19 domains and their rules
Run validation sample - process_enhanced_validation_sample() to see it work
Check YAML directory - See the individual category structure files
Examine checkpoints - Understand how large datasets are processed
Test problematic cases - Use web search reclassification for difficult products

Key Functions to Understand:

classify_product() - Main entry point for classification
select_domain_with_llm() - Stage 1 domain selection
_classify_within_domain() - Stage 2 deep classification
tag_product() - Stage 3 tagging
validate_classification_path() - Path validation against YAML



📚 Table of Contents

🎯 What This System Does
🏗️ Architecture Overview
⚡ Quick Start
📁 File Structure
🔄 How Classification Works
⚙️ Configuration
🚀 Usage Examples
🔍 Web Search Reclassification
💾 Checkpointing System
📊 Output Format
🛠️ Development Guide
❓ Troubleshooting


🎯 What This System Does
This system automatically classifies life science products (lab equipment, reagents, antibodies, kits, etc.) into:
✅ 19 specialized domains (Cell Biology, Lab Equipment, Antibodies, etc.)
✅ Deep hierarchical categories (up to 4 levels: Domain → Subcategory → Subsubcategory → Subsubsubcategory)
✅ 4 types of cross-cutting tags (Technique, Research, Functional, Specimen)
✅ Dual classifications for products that fit multiple domains
✅ Web search enhancement for difficult cases
Example Output:
Product: "Anti-NeuN monoclonal antibody"
Primary: Antibodies → Research Area Antibodies → Neuroscience Antibodies → Neuronal Marker Antibodies
Tags: Technique=[Immunofluorescence, Western Blot], Research=[Neuroscience], Functional=[Research Use Only], Specimen=[Human, Mouse]

🏗️ Architecture Overview
mermaidgraph TD
    A[Product Input] --> B[Stage 1: Domain Selection]
    B --> C[Stage 2a: Primary Classification]
    B --> D[Stage 2b: Secondary Classification]
    C --> E[Stage 3: Unified Tagging]
    D --> E
    E --> F[Final Output]
    
    G[Master Categories YAML] --> B
    H[Domain-Specific YAMLs] --> C
    H --> D
    I[Web Search] --> J[Reclassification]
    F --> K{Problematic?}
    K -->|Yes| I
    K -->|No| L[Complete]
    
Core Components
ComponentPurposeFileMaster ControllerMaps domains to YAML files, provides classification rulesmaster_categories_claude.yamlClassification Engine3-stage AI classification pipelineenhanced_classification_full_checkpoint.pyWeb Search ModuleReclassifies problematic cases using real web dataweb_search_integration.pyDomain StructuresDetailed hierarchical category treescategory_structure_*.yaml files

⚡ Quick Start
Prerequisites
bashpip install pandas openai pyyaml tqdm python-dotenv
Environment Setup

Create API key file: api_keys/OPEN_AI_KEY.env

envOPENAI_API_KEY=your_openai_api_key_here

Ensure YAML files are in place:

category_structures/
├── master_categories_claude.yaml
├── category_structure_cell_biology.yaml
├── category_structure_lab_equipment.yaml
└── ... (15 more domain YAML files)
Basic Usage
pythonfrom enhanced_classification_full_checkpoint import *

# Initialize the system
category_system = LLMDrivenCategorySystem()
tag_system = EnhancedTagSystem()
classifier = EnhancedLLMClassifier(category_system, tag_system)

# Classify a single product
result = classifier.classify_product(
    product_name="Anti-CD3 monoclonal antibody",
    description="Monoclonal antibody for flow cytometry"
)

print(f"Domain: {result['primary_classification']['domain']}")
print(f"Path: {result['primary_classification']['validated_path']}")
print(f"Tags: {result['primary_classification']['technique_tags']}")
Process Dataset
python# For small validation (100 products)
process_enhanced_validation_sample()

# For full dataset (50K+ products)  
process_full_dataset()

📁 File Structure
project/
├── enhanced_classification_full_checkpoint.py    # Main classification engine
├── web_search_integration.py                     # Web search reclassification
├── master_categories_claude.yaml                 # Master configuration
├── category_structures/                          # Domain-specific structures
│   ├── category_structure_cell_biology.yaml
│   ├── category_structure_lab_equipment.yaml
│   ├── category_structure_antibodies.yaml
│   └── ... (14 more)
├── api_keys/
│   └── OPEN_AI_KEY.env                           # API credentials
├── enhanced_classification_checkpoints/          # Progress saves
└── data/
    ├── input_products.csv                        # Source data
    └── products_enhanced_fixed_classification.csv # Output

🔄 How Classification Works
Stage 1: Smart Domain Selection 🎯
Input: Product name + description
Output: 1-2 best domains with confidence scores
AI Model: GPT-4o-mini
Key Rules:

Product Type beats Application - Microscopes go to Lab Equipment, not by what they analyze
Equipment vs Reagent Distinction - Physical instruments vs chemicals/biologicals
Dual Domain Detection - Some products legitimately fit 2 domains

Example:
pythonInput: "qPCR thermocycler system"
Output: Primary=Lab_Equipment (95% conf), Secondary=Molecular_Diagnostics (80% conf)
Stage 2: Deep Hierarchical Classification 📊
Input: Product + selected domain(s)
Output: Deep classification path (up to 4 levels)
AI Model: GPT-4o-mini
Process:

Load domain-specific YAML structure
Show AI ALL possible classification paths
AI selects deepest appropriate path
Validate against actual YAML structure

Example:
pythonDomain: Antibodies
Input: "Anti-NeuN monoclonal antibody"
Output: Research Area Antibodies → Neuroscience Antibodies → Neuronal Marker Antibodies → NeuN Antibodies
Stage 3: Cross-Cutting Tagging 🏷️
Input: Product + domain classification(s)
Output: 4 categories of tags
AI Model: GPT-4o-mini
Tag Categories:

Technique Tags: HOW it's used (Western Blot, Flow Cytometry, PCR)
Research Tags: WHAT research it supports (Cancer Research, Neuroscience)
Functional Tags: WHAT TYPE it is (Kit, Instrument, High-Throughput)
Specimen Tags: WHAT species it works with (Human, Mouse, Rat)


⚙️ Configuration
Master Categories File
The master_categories_claude.yaml is the control center:
yamldomain_mapping:
  Cell_Biology:
    yaml_file: "category_structure_cell_biology.yaml"
    description: "Cell culture systems, cell analysis tools..."
    keywords: ["cell", "culture", "staining", "flow cytometry"]
    typical_products: ["HeLa cells", "Giemsa stain", "Flow cytometry buffer"]
    classification_hints:
      - "Giemsa, crystal violet = staining reagents, NOT equipment"
      - "Cell culture stains have dedicated categories"
Domain-Specific YAMLs
Each domain has detailed hierarchical structure:
yaml# Example from category_structure_antibodies.yaml
Antibodies:
  subcategories:
    Primary Antibodies:
      subsubcategories:
        Monoclonal Antibodies:
          subsubsubcategories:
            - IgG1 Isotype Controls
            - IgG2a Isotype Controls
        Application-Specific Primary Antibodies:
          subsubsubcategories:
            - Western Blot Primary Antibodies
            - Flow Cytometry Primary Antibodies
Adding New Domains

Create new YAML file: category_structure_new_domain.yaml
Add entry to master_categories_claude.yaml:

yamlNew_Domain:
  yaml_file: "category_structure_new_domain.yaml"
  description: "Description of what this domain covers"
  keywords: ["keyword1", "keyword2"]
  typical_products: ["Example product 1", "Example product 2"]

🚀 Usage Examples
Classify Single Product
pythonfrom enhanced_classification_full_checkpoint import *

# Initialize
category_system = LLMDrivenCategorySystem()
tag_system = EnhancedTagSystem()
classifier = EnhancedLLMClassifier(category_system, tag_system)

# Classify
result = classifier.classify_product(
    "Human TNF-alpha ELISA Kit",
    "96-well quantitative ELISA for human TNF-alpha detection"
)

# Access results
primary = result['primary_classification']
print(f"Domain: {primary['domain']}")
print(f"Full Path: {primary['validated_path']}")
print(f"Confidence: {primary['confidence']}")
print(f"Technique Tags: {primary['technique_tags']}")
Process CSV File
pythonimport pandas as pd

# Load your data
df = pd.read_csv("your_products.csv")
# Required columns: Name, Description (optional), Manufacturer (optional)

# Process with checkpointing
results_df = process_full_dataset()

# Or test with sample
validation_df = process_enhanced_validation_sample()
Check Domain Availability
pythoncategory_system = LLMDrivenCategorySystem()
print("Available domains:")
for domain in category_system.available_domains:
    print(f"  - {domain}")

🔍 Web Search Reclassification
For products with problematic classifications, the system can use web search to get better information:
When It's Triggered

Products classified as "Other" domain
Generic classifications like "Analytical Instrumentation"
Misclassified antibiotics (e.g., streptomycin as acid instead of antibiotic)

How to Use
pythonfrom web_search_integration import *

# Process problematic products with web search
complete_df, problematic_df = process_corrected_real_web_search_reclassification_with_checkpointing(
    input_csv="products_enhanced_fixed_classification.csv",
    output_csv="products_real_web_search_reclassified.csv"
)
What It Does

Identifies problematic products from previous classification runs
Performs real web search using OpenAI's web search API
Gets enhanced product information from official sources
Re-classifies with better context
Updates original classifications if successful


💾 Checkpointing System
The system automatically saves progress to handle large datasets and API failures:
Checkpoint Types

Validation Checkpoints: enhanced_validation_checkpoint_N.csv
Full Dataset Checkpoints: enhanced_full_checkpoint_N.csv
Web Search Checkpoints: real_web_search_checkpoint_N.csv

How It Works
python# Saves every 50 products (configurable)
CHECKPOINT_FREQ = 50

# Auto-resumes from latest checkpoint
latest_checkpoint = find_latest_checkpoint()
if latest_checkpoint:
    df = pd.read_csv(latest_checkpoint)
    # Resume from where it left off
Manual Checkpoint Management
python# Find latest checkpoint
checkpoint_file = find_latest_checkpoint()
print(f"Latest checkpoint: {checkpoint_file}")

# Load specific checkpoint
df = pd.read_csv("enhanced_validation_checkpoint_150.csv")

📊 Output Format
CSV Columns Structure
Primary Classification
primary_domain                    # Main domain (e.g., "Antibodies")
primary_subcategory              # Level 1 (e.g., "Primary Antibodies") 
primary_subsubcategory           # Level 2 (e.g., "Monoclonal Antibodies")
primary_subsubsubcategory        # Level 3 (e.g., "IgG1 Isotype Controls")
primary_confidence               # High/Medium/Low
validated_path_primary           # Full path string
Secondary Classification (for dual-function products)
secondary_domain
secondary_subcategory
secondary_subsubcategory  
secondary_subsubsubcategory
secondary_confidence
validated_path_secondary
Tagging Results
technique_tags                   # Pipe-separated (e.g., "Flow Cytometry|Western Blot")
research_tags                    # Pipe-separated (e.g., "Immunology|Cancer Research")
functional_tags                  # Pipe-separated (e.g., "Research Use Only|High Specificity")
specimen_tags                    # Pipe-separated (e.g., "Human|Mouse")
total_tags                       # Count of all tags
tag_confidence                   # High/Medium/Low
Metadata
is_dual_function                 # True/False
classification_count             # 1 or 2
total_token_usage               # API cost tracking
error_occurred                  # True/False
Example Output Row
csvName,primary_domain,primary_subcategory,primary_subsubcategory,technique_tags,research_tags,total_tags
"Anti-CD3 antibody",Antibodies,Primary Antibodies,Monoclonal Antibodies,"Flow Cytometry|Western Blot","Immunology|Cell Biology",4

🛠️ Development Guide
Adding New Tag Categories
python# In EnhancedTagSystem class
class EnhancedTagSystem:
    def __init__(self):
        # Add new tag category
        self.new_category_tags = {
            "Tag1", "Tag2", "Tag3"
        }
        
        # Update all_tags
        self.all_tags = (
            self.technique_tags | 
            self.research_application_tags | 
            self.functional_tags | 
            self.specimen_tags |
            self.new_category_tags  # Add here
        )
Modifying Classification Rules
python# In master_categories_claude.yaml, add domain-specific rules:
Cell_Biology:
  classification_hints:
    - "New rule: Products with X should go to subcategory Y"
    - "Exception: Z products are special cases"
Custom Domain Intelligence
python# Add custom validation logic
def validate_custom_rules(self, product_name, domain, classification):
    if "special_keyword" in product_name.lower():
        # Apply custom logic
        return modified_classification
    return classification
Testing New Features
python# Test with small sample
def test_new_feature():
    test_products = [
        {"name": "Test Product 1", "description": "Test description"},
        {"name": "Test Product 2", "description": "Another test"}
    ]
    
    for product in test_products:
        result = classifier.classify_product(product["name"], product["description"])
        print(f"Result: {result}")

❓ Troubleshooting
Common Issues
🔴 "Domain not found" Error
python# Check available domains
category_system = LLMDrivenCategorySystem()
print(category_system.available_domains)

# Verify YAML files exist
import os
yaml_dir = "category_structures"
for domain in category_system.available_domains:
    yaml_file = f"category_structure_{domain.lower()}.yaml"
    path = os.path.join(yaml_dir, yaml_file)
    print(f"{domain}: {os.path.exists(path)}")
🔴 OpenAI API Errors
python# Check API key
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path("api_keys/OPEN_AI_KEY.env")
if env_path.exists():
    load_dotenv(env_path)
    key = os.getenv("OPENAI_API_KEY")
    print(f"API key loaded: {'Yes' if key else 'No'}")
else:
    print("API key file not found")
🔴 Memory Issues with Large Datasets
python# Reduce checkpoint frequency
CHECKPOINT_FREQ = 25  # Save more frequently

# Process in smaller batches
df_chunks = [df[i:i+1000] for i in range(0, len(df), 1000)]
for chunk in df_chunks:
    process_chunk(chunk)
🔴 Classification Quality Issues
python# Check validation report
validation_df = process_enhanced_validation_sample()
generate_enhanced_dual_validation_report(validation_df)

# Look for patterns in failed classifications
failed = validation_df[validation_df['primary_confidence'] == 'Low']
print(failed[['Name', 'primary_domain', 'primary_subcategory']])
Performance Optimization
Reduce Token Usage
python# Use shorter descriptions
max_description_length = 200
description = description[:max_description_length] if description else ""

# Optimize prompt templates (reduce example count)
tag_categories = self.tag_system.format_compact_prompt(max_per_category=8)  # Reduce from 15
Speed Up Processing
python# Parallel processing (experimental)
from concurrent.futures import ThreadPoolExecutor

def process_parallel(products, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(classify_single_product, products))
    return results
Getting Help

Check logs - All operations are logged with timestamps
Examine checkpoint files - See what was processed before failure
Run validation sample - Test system health with 100 products
Review master categories - Ensure domain rules make sense
Test single products - Isolate issues with specific examples


📈 Performance Metrics
Typical Performance

Classification Speed: ~2-3 products/minute
Token Usage: ~800-1200 tokens per product
Cost: ~$0.0001-0.0002 per product (GPT-4o-mini)
Accuracy: 85-90% correct domain selection
Depth: 70%+ achieve 3+ level classification

Quality Indicators

Valid Path Rate: 95%+
High Confidence Rate: 60-70%
Dual Classification Rate: 5-10%
Deep Classification Rate: 70%+ reach subsubcategory
