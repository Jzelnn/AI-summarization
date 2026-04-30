"""
Text Preprocessing Module for NLP Pipeline
Cleans and normalizes text before processing
"""

import re
import string
from typing import List


def clean_whitespace(text: str) -> str:
    """Remove extra whitespace and normalize spacing"""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text"""
    # Remove http/https URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove www URLs
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text


def remove_emails(text: str) -> str:
    """Remove email addresses"""
    text = re.sub(r'\S+@\S+', '', text)
    return text


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters
    
    Args:
        text: Input text
        keep_punctuation: If True, keep .!?,;: for sentence structure
    """
    if keep_punctuation:
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.!?,;:\'-]', '', text)
    else:
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize excessive punctuation"""
    # Remove multiple punctuation marks (e.g., "!!!" -> "!")
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Add space after punctuation if missing
    text = re.sub(r'([,.!?;:])([A-Za-z])', r'\1 \2', text)
    return text


def remove_extra_newlines(text: str) -> str:
    """Replace multiple newlines with single newline"""
    text = re.sub(r'\n+', '\n', text)
    return text


def preprocess_for_summarization(text: str) -> str:
    """
    Preprocessing for summarization
    Keep sentence structure and proper nouns
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text ready for summarization
    """
    # Remove URLs and emails (usually noise)
    text = remove_urls(text)
    text = remove_emails(text)
    
    # Remove extra newlines
    text = remove_extra_newlines(text)
    
    # Normalize punctuation
    text = normalize_punctuation(text)
    
    # Clean whitespace
    text = clean_whitespace(text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text


def preprocess_for_ner(text: str) -> str:
    """
    Preprocessing for Named Entity Recognition
    Keep capitalization - important for NER!
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text ready for NER
    """
    # Remove URLs (but keep the rest)
    text = remove_urls(text)
    
    # Normalize punctuation
    text = normalize_punctuation(text)
    
    # Clean whitespace
    text = clean_whitespace(text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text


def preprocess_for_topic_detection(text: str) -> str:
    """
    Preprocessing for topic detection
    More aggressive cleaning, lowercase for keyword matching
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text ready for topic detection
    """
    # Remove URLs and emails
    text = remove_urls(text)
    text = remove_emails(text)
    
    # Convert to lowercase (for keyword matching)
    text = text.lower()
    
    # Clean whitespace
    text = clean_whitespace(text)
    
    return text


def basic_preprocess(text: str) -> str:
    """
    Basic preprocessing for general use
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    # Remove URLs and emails
    text = remove_urls(text)
    text = remove_emails(text)
    
    # Remove extra newlines
    text = remove_extra_newlines(text)
    
    # Normalize punctuation
    text = normalize_punctuation(text)
    
    # Clean whitespace
    text = clean_whitespace(text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text


# Testing
if __name__ == "__main__":
    test_text = """
    Hello!!!   This is a   test text with    extra    spaces.
    
    
    Visit my website at https://example.com or email me at test@example.com
    
    I am from Jakarta, Indonesia. Tim Cook is the CEO of Apple Inc.
    """
    
    print("Original text:")
    print(repr(test_text))
    print("\n" + "="*60 + "\n")
    
    print("After basic preprocessing:")
    print(basic_preprocess(test_text))
    print("\n" + "="*60 + "\n")
    
    print("For summarization:")
    print(preprocess_for_summarization(test_text))
    print("\n" + "="*60 + "\n")
    
    print("For NER (keeps capitalization):")
    print(preprocess_for_ner(test_text))
    print("\n" + "="*60 + "\n")
    
    print("For topic detection (lowercase):")
    print(preprocess_for_topic_detection(test_text))