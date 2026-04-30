"""
Named Entity Recognition (NER) Module
This module extracts named entities from text using a fine-tuned spaCy model
"""

import spacy
from typing import Dict, List, Optional
from pathlib import Path

# Try to load the fine-tuned model first, fallback to base model
MODEL_PATH = Path("nlp/fine_tuned_ner")

try:
    if MODEL_PATH.exists():
        nlp = spacy.load(MODEL_PATH)
        print("✅ Loaded fine-tuned NER model")
    else:
        print("⚠️ Fine-tuned model not found, using base model")
        print("   Run fine_tune_ner.py first for better results!")
        nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Warning: spaCy model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str, use_filtering: bool = True) -> Dict[str, List[str]]:
    """
    Extract named entities from text using spaCy
    
    Args:
        text (str): Input text to extract entities from
        use_filtering (bool): Apply post-processing filters to improve accuracy
        
    Returns:
        Dict[str, List[str]]: Dictionary with entity types as keys and lists of entities as values
        
    Example:
        >>> extract_entities("Apple Inc. was founded by Steve Jobs")
        {
            'ORG': ['Apple Inc.'],
            'PERSON': ['Steve Jobs']
        }
    """
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize the entities dictionary
    entities = {}
    
    # Extract entities
    for ent in doc.ents:
        entity_type = ent.label_
        entity_text = ent.text.strip()
        
        # Skip empty entities
        if not entity_text:
            continue
        
        # Apply filtering if enabled
        if use_filtering and should_filter_entity(entity_text, entity_type, text):
            continue
        
        # Initialize the list for this entity type if it doesn't exist
        if entity_type not in entities:
            entities[entity_type] = []
        
        # Add entity if it's not already in the list (avoid duplicates)
        if entity_text not in entities[entity_type]:
            entities[entity_type].append(entity_text)
    
    # Debug logging
    print(f"🏷️ NER - Extracted {len(entities)} entity types:")
    for entity_type, items in entities.items():
        print(f"   - {entity_type}: {len(items)} items → {items}")
    
    return entities


def should_filter_entity(entity_text: str, entity_type: str, full_text: str) -> bool:
    """
    Filter out incorrectly classified entities
    
    Args:
        entity_text (str): The entity text
        entity_type (str): The classified entity type
        full_text (str): The full text for context
        
    Returns:
        bool: True if entity should be filtered out, False otherwise
    """
    entity_lower = entity_text.lower()
    
    # Common country/nationality names that should be TEAM or GPE, not PRODUCT
    countries_teams = {
        'argentina', 'brazil', 'germany', 'france', 'spain', 'england',
        'portugal', 'italy', 'netherlands', 'belgium', 'croatia',
        'mexico', 'usa', 'canada', 'japan', 'south korea', 'australia',
        'egypt', 'morocco', 'senegal', 'nigeria', 'ghana', 'cameroon',
        'uruguay', 'chile', 'colombia', 'poland', 'sweden', 'denmark'
    }
    
    # If a country is classified as PRODUCT, filter it out
    if entity_type == "PRODUCT" and entity_lower in countries_teams:
        return True
    
    # Common football/sports terms that shouldn't be organizations
    common_words = {
        'goal', 'match', 'game', 'ball', 'field', 'stadium',
        'team', 'player', 'coach', 'manager', 'win', 'loss'
    }
    
    if entity_lower in common_words:
        return True
    
    # Filter out single letters or very short non-name entities
    if len(entity_text) <= 1:
        return True
    
    return False


def get_entity_types() -> List[str]:
    """
    Get list of supported entity types in spaCy
    
    Returns:
        List[str]: List of entity type labels
    """
    base_types = [
        "PERSON",      # People, including fictional
        "NORP",        # Nationalities or religious or political groups
        "FAC",         # Buildings, airports, highways, bridges, etc.
        "ORG",         # Companies, agencies, institutions, etc.
        "GPE",         # Countries, cities, states
        "LOC",         # Non-GPE locations, mountain ranges, bodies of water
        "PRODUCT",     # Objects, vehicles, foods, etc. (not services)
        "EVENT",       # Named hurricanes, battles, wars, sports events, etc.
        "WORK_OF_ART", # Titles of books, songs, etc.
        "LAW",         # Named documents made into laws
        "LANGUAGE",    # Any named language
        "DATE",        # Absolute or relative dates or periods
        "TIME",        # Times smaller than a day
        "PERCENT",     # Percentage, including "%"
        "MONEY",       # Monetary values, including unit
        "QUANTITY",    # Measurements, as of weight or distance
        "ORDINAL",     # "first", "second", etc.
        "CARDINAL"     # Numerals that do not fall under another type
    ]
    
    # Add custom types if fine-tuned model is loaded
    custom_types = ["PLAYER", "TEAM"]
    
    return base_types + custom_types


def extract_entities_by_type(text: str, entity_types: List[str]) -> Dict[str, List[str]]:
    """
    Extract only specific types of entities from text
    
    Args:
        text (str): Input text
        entity_types (List[str]): List of entity types to extract (e.g., ["PERSON", "ORG"])
        
    Returns:
        Dict[str, List[str]]: Dictionary with only the specified entity types
    """
    all_entities = extract_entities(text)
    
    # Filter to only requested types
    filtered_entities = {
        entity_type: entities 
        for entity_type, entities in all_entities.items() 
        if entity_type in entity_types
    }
    
    return filtered_entities


def count_entities(text: str) -> Dict[str, int]:
    """
    Count the number of entities of each type
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, int]: Dictionary with entity types and their counts
    """
    entities = extract_entities(text)
    return {entity_type: len(items) for entity_type, items in entities.items()}


def extract_sports_entities(text: str) -> Dict[str, List[str]]:
    """
    Specialized function for extracting sports-related entities
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, List[str]]: Dictionary with PLAYER and TEAM entities
    """
    return extract_entities_by_type(text, ["PLAYER", "TEAM", "PERSON", "ORG", "GPE"])


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Lionel Messi scored a goal for Argentina",
        "Cristiano Ronaldo joined Al Nassr",
        "Manchester United defeated Liverpool 3-1 in the Premier League",
        "Neymar plays for Brazil national team",
        "Apple Inc. was founded by Steve Jobs in California",
        "The United Nations held a meeting in New York City on January 15, 2024.",
        "Kylian Mbappé and Erling Haaland are competing for the Golden Boot"
    ]
    
    print("="*80)
    print("TESTING NER MODULE")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}:")
        print(f"Text: {text}")
        print(f"\n🏷️  Entities:")
        entities = extract_entities(text)
        
        if not entities:
            print("  ⚠️ No entities found")
        else:
            for entity_type, items in entities.items():
                print(f"  {entity_type}:")
                for item in items:
                    print(f"    - {item}")
        
        print(f"\n📊 Counts: {count_entities(text)}")
        print("-"*80)
    
    # Test sports-specific extraction
    print("\n" + "="*80)
    print("TESTING SPORTS ENTITY EXTRACTION")
    print("="*80)
    
    sports_text = "Lionel Messi and Cristiano Ronaldo are legends. Messi played for Argentina and Barcelona."
    print(f"\nText: {sports_text}")
    print("\n🏆 Sports Entities:")
    sports_entities = extract_sports_entities(sports_text)
    
    for entity_type, items in sports_entities.items():
        if items:
            print(f"  {entity_type}: {items}")