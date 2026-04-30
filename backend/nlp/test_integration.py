"""
Enhanced NLP Pipeline with AI Summarization and Text Preprocessing
Supports both HuggingFace and Gemini backends
FIXED: Non-English NER issue - now translates to English before NER
FIXED: Import structure works both as module and standalone
"""

import os
import sys

# FIXED: Try both relative and absolute imports to work in all contexts
try:
    # Try relative imports first (when used as nlp.pipeline)
    from .ner import extract_entities
    from .preprocessing import (
        preprocess_for_summarization,
        preprocess_for_ner,
        preprocess_for_topic_detection,
        basic_preprocess
    )
except ImportError:
    # Fall back to absolute imports (when run standalone or from different location)
    try:
        from nlp.ner import extract_entities
        from nlp.preprocessing import (
            preprocess_for_summarization,
            preprocess_for_ner,
            preprocess_for_topic_detection,
            basic_preprocess
        )
    except ImportError:
        # Last resort: direct imports (when nlp folder is in path)
        from ner import extract_entities
        from preprocessing import (
            preprocess_for_summarization,
            preprocess_for_ner,
            preprocess_for_topic_detection,
            basic_preprocess
        )

# Configuration - Choose your AI backend
AI_BACKEND = os.getenv("AI_BACKEND", "gemini")  # Options: "huggingface", "gemini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this for Gemini

# Initialize the appropriate summarizer
summarize = None

if AI_BACKEND == "gemini":
    try:
        try:
            from .ai_summarizer_gemini import summarize
        except ImportError:
            try:
                from nlp.ai_summarizer_gemini import summarize
            except ImportError:
                from ai_summarizer_gemini import summarize
        print("✅ Using Gemini AI for summarization")
    except Exception as e:
        print(f"⚠️ Gemini initialization failed: {e}")
        print("Falling back to HuggingFace...")
        AI_BACKEND = "huggingface"

if AI_BACKEND == "huggingface" and summarize is None:
    try:
        try:
            from .ai_summarizer_huggingface import summarize
        except ImportError:
            try:
                from nlp.ai_summarizer_huggingface import summarize
            except ImportError:
                from ai_summarizer_huggingface import summarize
        print("✅ Using HuggingFace AI for summarization")
    except Exception as e:
        print(f"⚠️ HuggingFace initialization failed: {e}")
        print("Falling back to basic summarization...")

# Last fallback to basic summarizer
if summarize is None:
    try:
        try:
            from .summarizer import summarize
        except ImportError:
            try:
                from nlp.summarizer import summarize
            except ImportError:
                from summarizer import summarize
        print("⚠️ Using basic summarization (no AI)")
    except ImportError:
        print("❌ No summarizer available!")
        # Define a minimal fallback
        def summarize(text, level="medium", topic=None):
            sentences = text.split('.')
            n = max(1, len(sentences) // 3)
            return '. '.join(sentences[:n])

# FIXED: Import multilingual processor for NER translation
ml_processor = None
MULTILINGUAL_AVAILABLE = False

try:
    try:
        from .multilingual import MultilingualProcessor
    except ImportError:
        try:
            from nlp.multilingual import MultilingualProcessor
        except ImportError:
            from multilingual import MultilingualProcessor
    
    ml_processor = MultilingualProcessor()
    MULTILINGUAL_AVAILABLE = True
    print("✅ Multilingual support enabled for NER")
except Exception as e:
    print(f"⚠️ Multilingual support not available: {e}")


def detect_topic(text: str) -> str:
    """
    Detect the topic/category of the text
    
    Args:
        text: Input text (will be preprocessed internally)
        
    Returns:
        Detected topic: "sports", "politics", "tech", or "general"
    """
    # Preprocess text for topic detection (lowercase, cleaned)
    text_processed = preprocess_for_topic_detection(text)
    
    # Define keywords for each topic
    topic_keywords = {
        "sports": ["match", "goal", "player", "football", "score", "team", "game", 
                  "championship", "tournament", "coach", "athlete", "soccer", "basketball",
                  "cricket", "tennis", "stadium", "league", "champion", "trophy"],
        "politics": ["election", "government", "president", "policy", "parliament", 
                    "congress", "senator", "politician", "vote", "legislation", "political",
                    "minister", "prime minister", "democracy", "senate", "bill", "law"],
        "tech": ["technology", "ai", "software", "computer", "data", "system", "digital",
                "algorithm", "programming", "innovation", "startup", "tech", "artificial",
                "intelligence", "machine learning", "cloud", "app", "application"]
    }
    
    # Score each topic
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_processed)
        topic_scores[topic] = score
    
    # Find the topic with the highest score
    max_score = max(topic_scores.values())
    
    if max_score == 0:
        return "general"
    
    detected_topic = max(topic_scores, key=topic_scores.get)
    
    print(f"🏷️ Detected topic: {detected_topic} (confidence: {max_score} keywords)")
    return detected_topic


def extract_entities_multilingual(text: str, detected_lang: str = None) -> dict:
    """
    FIXED: Extract entities from text, translating to English if needed
    
    Args:
        text: Input text in any language
        detected_lang: Already detected language (optional)
        
    Returns:
        Dictionary of entities by type
    """
    # Preprocess text for NER
    text_cleaned = basic_preprocess(text)
    text_for_ner = preprocess_for_ner(text_cleaned)
    
    # FIXED: Detect language if not provided
    if ml_processor and MULTILINGUAL_AVAILABLE:
        if detected_lang is None:
            detected_lang = ml_processor.detect_language(text_for_ner)
        
        # If not English, translate to English for NER
        if detected_lang and detected_lang != 'en':
            print(f"🌐 Translating from {detected_lang} to English for accurate NER...")
            try:
                text_for_ner = ml_processor.translate_text(
                    text_for_ner,
                    target_lang='en',
                    source_lang=detected_lang
                )
                print(f"✅ Translated text for NER: {text_for_ner[:100]}...")
            except Exception as e:
                print(f"⚠️ Translation failed: {e}. Using original text.")
    
    # Now extract entities from English text
    return extract_entities(text_for_ner)


def full_pipeline(text: str, length: str = "medium", detected_lang: str = None) -> dict:
    """
    Full NLP pipeline with preprocessing: topic detection, summarization, and NER
    FIXED: Now handles multilingual NER properly
    
    Args:
        text: Raw input text to process
        length: Summary length - "short", "medium", or "long"
        detected_lang: Pre-detected language (optional)
        
    Returns:
        Dictionary containing:
            - detected_language: Detected language of input
            - topic: Detected topic
            - summary: AI-generated summary
            - entities: Named entities extracted from text
    """
    print(f"\n{'='*60}")
    print(f"🚀 Running Full NLP Pipeline with Preprocessing")
    print(f"{'='*60}")
    
    # Step 0: Basic preprocessing for all tasks
    print(f"🧹 Preprocessing input text...")
    text_cleaned = basic_preprocess(text)
    print(f"   Original length: {len(text)} chars")
    print(f"   After cleaning: {len(text_cleaned)} chars")
    
    # FIXED: Detect language
    if ml_processor and MULTILINGUAL_AVAILABLE and detected_lang is None:
        detected_lang = ml_processor.detect_language(text_cleaned)
        print(f"🌐 Detected language: {detected_lang}")
    
    # Step 1: Detect topic (uses its own preprocessing)
    print(f"🔍 Detecting topic...")
    topic = detect_topic(text_cleaned)
    
    # Step 2: Generate AI summary (preprocess for summarization)
    print(f"📝 Generating {length} summary...")
    text_for_summary = preprocess_for_summarization(text_cleaned)
    summary = summarize(text_for_summary, level=length, topic=topic)
    
    # Step 3: Extract named entities (FIXED: with multilingual support)
    print(f"🏷️ Extracting named entities...")
    entities = extract_entities_multilingual(text_cleaned, detected_lang)
    
    print(f"{'='*60}")
    print(f"✅ Pipeline completed successfully")
    print(f"   Language: {detected_lang or 'unknown'}")
    print(f"   Topic: {topic}")
    print(f"   Summary length: {len(summary)} chars")
    print(f"   Entities found: {sum(len(v) for v in entities.values())}")
    print(f"{'='*60}\n")
    
    return {
        "detected_language": detected_lang,
        "topic": topic,
        "summary": summary,
        "entities": entities
    }


def summarize_only(text: str, length: str = "medium", topic: str = None) -> str:
    """
    Direct summarization with preprocessing
    
    Args:
        text: Raw input text
        length: Summary length
        topic: Optional pre-detected topic
        
    Returns:
        Summary text
    """
    # Preprocess text
    text_cleaned = basic_preprocess(text)
    text_for_summary = preprocess_for_summarization(text_cleaned)
    
    if topic is None:
        topic = detect_topic(text_cleaned)
    
    return summarize(text_for_summary, level=length, topic=topic)


def extract_entities_only(text: str, detected_lang: str = None) -> dict:
    """
    FIXED: Direct NER with preprocessing and multilingual support
    
    Args:
        text: Raw input text
        detected_lang: Pre-detected language (optional)
        
    Returns:
        Dictionary of entities by type
    """
    return extract_entities_multilingual(text, detected_lang)


# Testing
if __name__ == "__main__":
    # Test with English text
    english_text = """
    Lionel Messi scored a stunning goal last night as Argentina defeated Brazil 2-1 
    in a thrilling Copa America final. The match was held in Buenos Aires and watched 
    by over 80,000 fans at the stadium. Messi once again proved why he is considered 
    one of the greatest players of all time.
    """
    
    # Test with Indonesian text
    indonesian_text = """
    Lionel Messi mencetak gol yang menakjubkan tadi malam ketika Argentina mengalahkan 
    Brasil 2-1 dalam final Copa America yang menegangkan. Pertandingan diadakan di 
    Buenos Aires dan disaksikan oleh lebih dari 80.000 penggemar di stadion.
    """
    
    print("="*80)
    print("TESTING FULL PIPELINE WITH MULTILINGUAL NER (FIXED)")
    print("="*80)
    
    print("\n📝 TEST 1: English Text")
    print("-"*80)
    result_en = full_pipeline(english_text, "medium")
    
    print("\n📊 RESULTS:")
    print(f"Language: {result_en['detected_language']}")
    print(f"Topic: {result_en['topic']}")
    print(f"\n📝 Summary:\n{result_en['summary']}")
    print(f"\n📖 Entities:")
    for entity_type, items in result_en['entities'].items():
        if items:
            print(f"   {entity_type.upper()}: {', '.join(items)}")
    
    if MULTILINGUAL_AVAILABLE:
        print("\n\n" + "="*80)
        print("📝 TEST 2: Indonesian Text (will be translated for NER)")
        print("-"*80)
        result_id = full_pipeline(indonesian_text, "medium")
        
        print("\n📊 RESULTS:")
        print(f"Language: {result_id['detected_language']}")
        print(f"Topic: {result_id['topic']}")
        print(f"\n📝 Summary:\n{result_id['summary']}")
        print(f"\n📖 Entities (extracted from English translation):")
        for entity_type, items in result_id['entities'].items():
            if items:
                print(f"   {entity_type.upper()}: {', '.join(items)}")
    
    print("\n" + "="*80)
    print("✅ Test completed!")
    print("="*80)