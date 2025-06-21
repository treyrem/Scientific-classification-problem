import pandas as pd
from typing import Dict, List, Pattern
import re
from functools import lru_cache

# Create category rules based on keywords and patterns
CATEGORY_RULES = {
    # Priority categories (most specific matches first)
    # Priority categories (most specific matches first)
    "Blotting Reagents": [
        r"transfer buffer",
        r"blocking buffer",
        r"membrane",
        r"film",
    ],
    # Growth Factors & Cytokines (includes substrates)
    "Growth Factors & Cytokines": [
        r"\begf\b",
        r"\bfgf\b",
        r"\bil-\d+\b",
        r"\btnf\b",
        r"\bgrowth factor\b",
        r"\bcytokine\b",
        r"substrate",
    ],
    "Antibodies": [
        r"\bantibody\b",
        r"ab\d+",  # Keep as is since it's a specific pattern
        r"\bimmunoglobulin\b",
        r"\banti-.*\b",
        r"\bprimary antibody\b",
        r"\bsecondary antibody\b",
    ],
    "Flow Cytometry": [
        r"\bflow cytometry\b",
        r"\bfacs\b",
        r"\bviability dye\b",
        r"\bfixative\b",
        r"\bflow cytometer\b",
        r"\bflow cytometric\b",
    ],
    "Genomics & Sequencing": [
        r"\blibrary prep\b",
        r"\bsequencing\b",
        r"\bngs\b",
        r"\billumina\b",
        r"\bpacbio\b",
        r"\bnanopore\b",
        r"\badapter\b",
        r"\bamplicon\b",
        r"\bprimer\b",
    ],
    "Oligonucleotides & Nucleotides": [
        r"\bprimer\b",
        r"\boligonucleotide\b",
        r"\bdntp\b",
        r"\bnucleotide\b",
        r"\bpcr\b",
        r"\bqpcr\b",
    ],
    "Probes & Markers": [
        r"\bLadder\b",
        r"\bmarker\b",
        r"\bprobe\b",
        r"\bloading dye\b",
        r"\bloading buffer\b",
    ],
    "Enzymes": [
        r"\benzyme\b",
        r"\bpolymerase\b",
        r"\bligase\b",
        r"\brestriction enzyme\b",
        r"\bendonuclease\b",
        r"\bkinase\b",
        r"\bproteinase\b",
    ],
    "Buffers & Solutions": [
        r"\btris\b",
        r"\bpbs\b",
        r"\btbs\b",
        r"\bte buffer\b",
        r"\bsds\b",
        r"\btween\b",
        r"\bglycerol\b",
        r"\bsolution\b",
    ],
    "Chemicals & Reagents": [
        r"\bbuffer\b",
        r"\bsolution\b",
        r"\breagent\b",
        r"\bchemical\b",
        r"\bacid\b",
        r"\bbase\b",
        r"\bsalt\b",
        r"\bmercaptoethanol\b",
        r"\bformaldehyde\b",
        r"\bacetonitrile\b",
        r"\bformic acid\b",
        r"\bhydrochloric acid\b",
        r"\bhcl\b",
    ],
    "Assay Kits": [
        r"\bassay kit\b",
        r"\btest kit\b",
        r"\bdetection kit\b",
        r"\belisa\b",
        r"\bqPCR master mix\b",
        r"\bmaster mix\b",
        r"\bkit\b",
        r"\bassay\b",
    ],
    "Nucleic Acid Purification": [
        r"\bpurification kit\b",
        r"\bextraction kit\b",
        r"\bisolation kit\b",
        r"\btotal rna\b",
        r"\btotal dna\b",
        r"\brna purification\b",
        r"\bdna purification\b",
        r"\bkit\b",
        r"\bextraction\b",
        r"\bpurification\b",
    ],
    "Chromatography": [
        r"\bchromatography\b",
        r"\bhplc\b",
        r"\bgc\b",
        r"\bfplc\b",
        r"\bcolumn\b",
        r"\bsepharose\b",
        r"\bsuperdex\b",
        r"\bc18\b",
        r"\bsize exclusion\b",
        r"\bion exchange\b",
    ],
    "Electrophoresis": [
        r"\bagarose\b",
        r"\bpolyacrylamide\b",
        r"\belectrophoresis\b",
        r"\bgel electrophoresis\b",
        r"\bpower supply\b",
        r"\bgel\b",
    ],
    "Proteomics": [
        r"\bmass spec\b",
        r"\bproteomics\b",
        r"\bcalibration standard\b",
        r"\bprotein ladder\b",
        r"\bprotein\b",
        r"\bpeptide\b",
    ],
    "Cell Culture Media & Reagents": [
        r"\bmedia\b",
        r"\bserum\b",
        r"\bfbs\b",
        r"\bdmem\b",
        r"\brpmi\b",
        r"\bglutamine\b",
        r"\bpyruvate\b",
        r"\bamino acids\b",
        r"\bculture media\b",
    ],
    "Microorganisms & Cells": [
        r"\bcell line\b",
        r"\bcell culture\b",
        r"\bbacteria\b",
        r"\byeast\b",
        r"\bfungi\b",
        r"\bmicroorganism\b",
        r"\bcell\b",
    ],
    "Microscopy & Imaging": [
        r"\bmicroscope\b",
        r"\bimaging\b",
        r"\bfluorescence\b",
        r"\bconfocal\b",
        r"\bbrightfield\b",
        r"\bdic\b",
        r"\bcamera\b",
        r"\bimager\b",
        r"\bchemidoc\b",
        r"\bscanner\b",
        r"\bplate reader\b",
        r"\bmultimode reader\b",
    ],
    "Laboratory Equipment": [
        r"\bcentrifuge\b",
        r"\bspectrophotometer\b",
        r"\bthermocycler\b",
        r"\bcycler\b",
        r"\bincubator\b",
        r"\bshaker\b",
        r"\bautoclave\b",
        r"\bfume hood\b",
    ],
    "Glassware": [
        r"\bbeaker\b",
        r"\bflask\b",
        r"\bbottle\b",
        r"\bslide\b",
        r"\bcoverglass\b",
        r"\bglass\b",
        r"\bvessel\b",
    ],
    "Plates & Dishes": [
        r"\bpetri dish\b",
        r"\bculture dish\b",
        r"\bmicroplate\b",
        r"\b96-?well\b",
        r"\bdish\b",
        r"\bplate\b",
    ],
    "Filters & Membranes": [
        r"\bfilter unit\b",
        r"\bsyringe filter\b",
        r"\bmembrane filter\b",
        r"\bfilter\b",
        r"\bmembrane\b",
    ],
    "Safety & Waste Disposal": [
        r"\bglove\b",
        r"\bppe\b",
        r"\bbiohazard\b",
        r"\bsharps\b",
        r"\bwaste disposal\b",
        r"\bsafety\b",
        r"\bprotection\b",
    ],
    "Life Science Software": [
        r"\bprism\b",
        r"\bspss\b",
        r"\bmatlab\b",
        r"\bimagequant\b",
        r"\boriginlab\b",  # updated regex pattern
        r"\bigor\b",
        r"\bgraphpad\b",
        r"\bcellprofiler\b",
        r"\bsoftware\b",
        r"\bprogram\b",
    ],
    "Dyes & Stains": [
        r"\bdapi\b",
        r"\bcoomassie\b",
        r"\bethidium\b",
        r"\bsybr\b",
        r"\bponceau\b",
        r"\bfluorescent dye\b",
        r"\bstain\b",
    ],
    "Growth Factors & Cytokines": [
        r"\begf\b",
        r"\bfgf\b",
        r"\bil-\d+\b",
        r"\btnf\b",
        r"\bgrowth factor\b",
        r"\bcytokine\b",
        r"substrate",
    ],

    # Laboratory glassware
    "Glassware": [
        r"beaker",
        r"flask",
        r"bottle",
        r"slide",
        r"coverglass",
    ],

    # Plates, dishes, and cultureware
    "Plates & Dishes": [
        r"petri dish",
        r"culture dish",
        r"microplate",
        r"96-?well",
    ],

    # Filter units and membrane filters
    "Filters & Membranes": [
        r"filter unit",
        r"syringe filter",
        r"membrane filter",
    ],

    # Safety equipment and waste disposal
    "Safety & Waste Disposal": [
        r"glove",
        r"ppe",
        r"biohazard",
        r"sharps",
        r"waste disposal",
    ],

    # Life science software tools
    "Life Science Software": [
        r"prism",
        r"spss",
        r"matlab",
        r"imagequant",
        r"origin",
        r"igor",
        r"graphpad",
        r"cellprofiler",
    ],

    # Fallback category
    "Other": []
}

# Pre-compile all regex patterns
COMPILED_RULES = {}
for category, rules in CATEGORY_RULES.items():
    COMPILED_RULES[category] = [re.compile(rule) for rule in rules if rule]  # Skip empty rules

@lru_cache(maxsize=1000)
def categorize_product(product_name: str) -> str:
    """Categorize a product based on its name using the pre-compiled regex patterns."""
    # Convert to lowercase for case-insensitive matching
    name = product_name.lower()
    
    # Check each category's compiled rules in priority order
    for category, patterns in COMPILED_RULES.items():
        # Skip empty patterns
        if not patterns:
            continue
        
        # Check if any pattern matches
        if any(pattern.search(name) for pattern in patterns):
            return category
    
    # If no specific category matches, return "Other"
    return "Other"

def main():
    # Read input CSV
    input_df = pd.read_csv('products_over_150_protocols_list.csv')
    
    # Create a new column for categories
    input_df['category'] = input_df['Name'].apply(categorize_product)
    
    # Save the categorized results
    input_df.to_csv('products_with_correct_categories.csv', index=False)
    print(f"Categorized {len(input_df)} products and saved to 'products_with_correct_categories.csv'")

if __name__ == "__main__":
    main()
