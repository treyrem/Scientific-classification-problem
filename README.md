This project aims to solve the complex task of lab product classification for unstructured list of 50k lab products using the Chatgpt 4-o-mini as an LLM classifier for multiclass classification problem in the absence of any datasets or labelled data to train an actual model.
Additionally, due to the formatting of lab product names in the original dataset, it required specialised domain knowledge to understand what the product is (some names were catalogue numbers)


Lab Product Categorization: A Tale of Trials, Errors, and Occasional Triumphs

Welcome to the cringy chronicles of my attempts to categorize 10,000 hyper-specific in places not specific enough lab product names into "right categories" (or somewhat reasonable). Spoiler alert: it wasn’t pretty—but hey, at least it was educational.

🏁 Chapter 1: The Naïve Debut (main.py)

Armed with nothing but misplaced confidence, claude ai and GPT-3.5-turbo API, I threw main.py at the problem. I fed it a list of 35 life-science categories and prayed it wouldn’t slap everything into Other.

Method: Prompt the LLM with a flat list of categories, one shot per product.

Reality Check: Only ~3,000 of the 10,000 names got a real label. The other ~7,000 were banished to Other, aka failed.

Downsides:

Massive Other Herd: When in doubt, GPT piled it into Other.  

Zero Nuance: A single layer of categories is like using a spoon to eat steak—technically possible but deeply unsatisfying.

Token Limit SOS: Prompt length hit the roof, so I had to butcher my examples down to the bare bones.

🚀 Chapter 2: Humble Pie & Regex Band‑Aids (main_2.py + product_categorizer.py)

After sobering up from Chapter 1’s meltdown, I tried a two-step method:

main_2.py: Upgraded to a two-tier taxonomy (10 big buckets + subcategories), still in LLM hands. 

product_categorizer.py: When AI still bailed, I lunged in with regex rules to patch the holes.

Progress (kind of):

Regex Wins: Uncategorizables shrank from ~7,000 to ~700. Victory? Sort of.

Structured-ish: Two levels helped a bit—some categories even made sense.

Still Awful:

700 OTHERS.

15% Oops Moments: My “Mass Spec Reagent” category included Mass Spec itself for some reason. And “Cell Lines” occasionally meant live mice and rats.

🌟 Chapter 3: The (Semi-)Glorious Comeback (new_categorization.py)

At this point, I desperately needed a win. Enter new_categorization.py, the crown jewel I’m only moderately proud of:

Thorough categorization analysis of the market. Surely smarter people from big lab equipment companies made better categories than I can possibly think of. 

Three Tiers: Category → Subcategory → Sub-subcategory. Because if two levels aren’t enough, three will surely do the trick.

JSON-Strict Prompts: LLM spits out JSON. No more guessing if it meant “Protein” or “Proteins.”

Fallbacks That Don’t Scream for Help: Invalid entries auto-correct to the closest valid option.

Regex Only Where Needed: A lean set of overrides for those brand-specific oddballs.

Checkpointing & Logs: Finally, I can pause, check my shitty checkpoints, and resume without starting from scratch.

Honarable Mentions:

Strugglebussers? Practically Gone: Under 1% of names remain mischievous. 

Error Rate: Single-digit misclassifications 7%. Let’s call it “good enough.”

FINAL VERSION:
The final version of classification was a glorius success with over 90 lab product categories and hierarchy established, the agent also evaluated the degree of confidence of its answers , which significantly  reduced the number of manual corrections done. From 7% missclassification to less than 1.5% missclassification for a dataset of 50000 Lab products.

🛠️ How to Torture-Test It Yourself if you ever end up in an unfortunate sutiation like mine. 

Clone or cd into the project.

pip install pandas openai tqdm python-dotenv

Add your OpenAI key to OPENAI_API_KEY.

python new_categorization.py

Celebrate small wins by opening products_with_categories_Claude.csv.

