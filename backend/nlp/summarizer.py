import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import re

TOPIC_KEYWORDS = {
    "sports": ["match", "goal", "player", "team", "football", "score"],
    "politics": ["government", "election", "president", "policy", "parliament"],
    "tech": ["technology", "ai", "software", "computer", "data", "system"]
}

nltk.download('punkt', quiet=True)

def summarize(text: str, level: str = "medium", topic: str | None = None) -> str:
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    words = re.findall(r'\w+', text.lower())
    freq = Counter(words)

    scores = {}
    for sent in sentences:
        score = 0
        sent_words = re.findall(r'\w+', sent.lower())

        for word in sent_words:
            score += freq.get(word, 0)

        # 🔥 BOOST TOPIC
        if topic in TOPIC_KEYWORDS:
            for kw in TOPIC_KEYWORDS[topic]:
                if kw in sent.lower():
                    score += 5

        scores[sent] = score

    if level == "short":
        n = max(1, len(sentences) // 4)
    elif level == "long":
        n = max(1, len(sentences) // 2)
    else:
        n = max(1, len(sentences) // 3)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return " ".join(ranked[:n])
