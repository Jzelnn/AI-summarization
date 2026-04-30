from nlp.summarizer import summarize
from nlp.ner import extract_entities

text = """
Lionel Messi scored a stunning goal last night as Argentina defeated Brazil.
The match was held in Buenos Aires and watched by thousands of fans.
Messi once again proved why he is considered one of the greatest players.
"""

summary = summarize(text, "medium")
entities = extract_entities(text)

print("SUMMARY:\n", summary)
print("\nENTITIES:")
for e in entities:
    print(e)
