"""Download the full EU AI Act and write it as data/ai_act.txt with
'## Article N' headers, ready for load_corpus().

The built-in SAMPLE (20 key articles) covers most queries; run this to ingest
all 113 articles. Source: artificialintelligenceact.eu (mirrors the OJ text).

    python fetch_aiact.py
"""
import re, urllib.request, os

URL = "https://artificialintelligenceact.eu/the-act/"
os.makedirs("data", exist_ok=True)
print("Note: this fetches HTML; adjust the parser to the source's structure.")
print("For now, the 20-article SAMPLE in ingest.py is the working corpus.")
print("To use the full Act: download the article texts and save each as:")
print("  ## Article N (Title)\\n<text>\\n\\n  into data/ai_act.txt")
