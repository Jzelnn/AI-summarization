"""
Multilingual Support for NLP Summarizer
Supports text translation and language detection for multilingual summarization
FIXED: Using deep-translator instead of googletrans to avoid httpx conflicts
"""

from typing import Literal, Optional, Dict, List
import re

# Try to import translation libraries
TRANSLATION_AVAILABLE = False
LANGDETECT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
    print("✅ Google Translate available (via deep-translator)")
except ImportError:
    print("⚠️ deep-translator not installed. Install with: pip install deep-translator")

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
    print("✅ Language detection available")
except ImportError:
    print("⚠️ langdetect not installed. Install with: pip install langdetect")


# Supported languages configuration
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'emoji': '🇺🇸'},
    'id': {'name': 'Indonesian', 'emoji': '🇮🇩'},
    'es': {'name': 'Spanish', 'emoji': '🇪🇸'},
    'fr': {'name': 'French', 'emoji': '🇫🇷'},
    'de': {'name': 'German', 'emoji': '🇩🇪'},
    'ja': {'name': 'Japanese', 'emoji': '🇯🇵'},
    'zh-CN': {'name': 'Chinese (Simplified)', 'emoji': '🇨🇳'},
    'zh-TW': {'name': 'Chinese (Traditional)', 'emoji': '🇹🇼'},
    'ko': {'name': 'Korean', 'emoji': '🇰🇷'},
    'ar': {'name': 'Arabic', 'emoji': '🇸🇦'},
    'ru': {'name': 'Russian', 'emoji': '🇷🇺'},
    'pt': {'name': 'Portuguese', 'emoji': '🇧🇷'},
    'it': {'name': 'Italian', 'emoji': '🇮🇹'},
    'nl': {'name': 'Dutch', 'emoji': '🇳🇱'},
    'tr': {'name': 'Turkish', 'emoji': '🇹🇷'},
    'vi': {'name': 'Vietnamese', 'emoji': '🇻🇳'},
    'th': {'name': 'Thai', 'emoji': '🇹🇭'},
    'hi': {'name': 'Hindi', 'emoji': '🇮🇳'},
}


class MultilingualProcessor:
    """Handle multilingual text processing and translation"""
    
    def __init__(self):
        """Initialize translator"""
        self.translator_available = TRANSLATION_AVAILABLE
        if TRANSLATION_AVAILABLE:
            print("✅ Translator initialized successfully")
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of input text
        
        Args:
            text: Input text
            
        Returns:
            ISO language code (e.g., 'en', 'id', 'es') or None if detection fails
        """
        if not LANGDETECT_AVAILABLE:
            print("⚠️ Language detection not available")
            return None
        
        try:
            # Clean text for better detection
            text_clean = re.sub(r'[^\w\s]', '', text)
            lang = detect(text_clean)
            
            confidence = detect_langs(text_clean)[0]
            print(f"🌐 Detected language: {lang} (confidence: {confidence.prob:.2%})")
            
            return lang
        except Exception as e:
            print(f"⚠️ Language detection failed: {e}")
            return None
    
    def translate_text(
        self,
        text: str,
        target_lang: str = 'en',
        source_lang: str = 'auto'
    ) -> str:
        """
        Translate text to target language
        FIXED: Using deep-translator instead of googletrans
        
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'en', 'id')
            source_lang: Source language code ('auto' for auto-detection)
            
        Returns:
            Translated text
        """
        if not self.translator_available:
            print("⚠️ Translation not available. Returning original text.")
            return text
        
        try:
            if source_lang == 'auto':
                source_lang = self.detect_language(text) or 'en'
            
            # Don't translate if already in target language
            if source_lang == target_lang:
                print(f"ℹ️ Text already in {target_lang}")
                return text
            
            print(f"🔄 Translating from {source_lang} to {target_lang}...")
            
            # FIXED: Use deep-translator (no httpx conflicts!)
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            
            # For long texts, split into chunks (deep-translator has 5000 char limit)
            if len(text) > 4500:
                print("📄 Text is long, translating in chunks...")
                chunks = self._split_into_chunks(text, max_length=4500)
                translated_chunks = []
                
                for i, chunk in enumerate(chunks, 1):
                    print(f"   Translating chunk {i}/{len(chunks)}...")
                    translated = translator.translate(chunk)
                    translated_chunks.append(translated)
                
                translated_text = " ".join(translated_chunks)
            else:
                translated_text = translator.translate(text)
            
            print(f"✅ Translation completed")
            return translated_text
            
        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            # Return original text on error
            return text
    
    def _split_into_chunks(self, text: str, max_length: int = 4500) -> List[str]:
        """
        Split text into chunks for translation
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Single sentence too long, just add it
                    chunks.append(sentence)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def multilingual_summarize(
        self,
        text: str,
        summarizer_func,
        input_lang: str = 'auto',
        output_lang: str = None,
        **summarizer_kwargs
    ) -> Dict[str, str]:
        """
        Summarize text with multilingual support
        
        Workflow:
        1. Detect input language
        2. Translate to English (if needed) for summarization
        3. Generate summary
        4. Translate summary back to desired output language
        
        Args:
            text: Input text in any language
            summarizer_func: Summarization function to use
            input_lang: Input language ('auto' for auto-detection)
            output_lang: Desired output language (None = same as input)
            **summarizer_kwargs: Additional arguments for summarizer
            
        Returns:
            Dictionary with:
                - original_text: Original input
                - detected_lang: Detected language code
                - text_for_summary: Text used for summarization (English)
                - summary_en: Summary in English
                - summary: Final summary in output language
                - output_lang: Final output language
        """
        # Step 1: Detect language
        if input_lang == 'auto':
            detected_lang = self.detect_language(text) or 'en'
        else:
            detected_lang = input_lang
        
        # Step 2: Translate to English for summarization (if needed)
        if detected_lang != 'en':
            print(f"🔍 Translating input to English for summarization...")
            text_en = self.translate_text(text, target_lang='en', source_lang=detected_lang)
        else:
            text_en = text
        
        # Step 3: Generate summary in English
        print(f"🤖 Generating summary...")
        summary_en = summarizer_func(text_en, **summarizer_kwargs)
        
        # Step 4: Determine output language
        if output_lang is None:
            output_lang = detected_lang
        
        # Step 5: Translate summary back if needed
        if output_lang != 'en':
            print(f"🔄 Translating summary to {output_lang}...")
            summary_final = self.translate_text(
                summary_en,
                target_lang=output_lang,
                source_lang='en'
            )
        else:
            summary_final = summary_en
        
        return {
            'original_text': text,
            'detected_lang': detected_lang,
            'text_for_summary': text_en,
            'summary_en': summary_en,
            'summary': summary_final,
            'output_lang': output_lang
        }
    
    def get_language_info(self, lang_code: str) -> Dict[str, str]:
        """Get information about a language"""
        return SUPPORTED_LANGUAGES.get(
            lang_code,
            {'name': 'Unknown', 'emoji': '🌐'}
        )
    
    def list_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of all supported languages"""
        return [
            {
                'code': code,
                'name': info['name'],
                'emoji': info['emoji']
            }
            for code, info in SUPPORTED_LANGUAGES.items()
        ]


# Simple fallback for basic language detection
def simple_detect_language(text: str) -> str:
    """
    Very simple language detection based on character patterns
    Fallback when langdetect is not available
    """
    # Check for CJK characters (Chinese, Japanese, Korean)
    if re.search(r'[\u4e00-\u9fff]', text):  # Chinese
        return 'zh-CN'
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese
        return 'ja'
    if re.search(r'[\uac00-\ud7af]', text):  # Korean
        return 'ko'
    
    # Check for Arabic
    if re.search(r'[\u0600-\u06ff]', text):
        return 'ar'
    
    # Check for Cyrillic (Russian)
    if re.search(r'[\u0400-\u04ff]', text):
        return 'ru'
    
    # Check for common Indonesian words
    indonesian_words = ['yang', 'dan', 'untuk', 'dari', 'dengan', 'pada', 'adalah']
    if any(word in text.lower() for word in indonesian_words):
        return 'id'
    
    # Default to English
    return 'en'


# Testing
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MULTILINGUAL PROCESSOR (FIXED - NO HTTPX CONFLICTS)")
    print("=" * 80)
    
    processor = MultilingualProcessor()
    
    # Test texts in different languages
    test_texts = {
        'English': "Artificial intelligence is transforming the world of technology.",
        'Indonesian': "Kecerdasan buatan mengubah dunia teknologi.",
        'Spanish': "La inteligencia artificial está transformando el mundo de la tecnología.",
        'Japanese': "人工知能は技術の世界を変革しています。",
        'Chinese': "人工智能正在改变技术世界。"
    }
    
    print("\n🌐 Testing Language Detection:")
    print("-" * 80)
    for lang_name, text in test_texts.items():
        detected = processor.detect_language(text)
        info = processor.get_language_info(detected or 'en')
        print(f"{lang_name}: {detected} {info['emoji']} - {info['name']}")
    
    print("\n\n🔄 Testing Translation:")
    print("-" * 80)
    
    if processor.translator_available:
        # Translate Indonesian to English
        id_text = "Lionel Messi mencetak gol yang menakjubkan untuk Argentina."
        en_translation = processor.translate_text(id_text, target_lang='en', source_lang='id')
        print(f"\nIndonesian: {id_text}")
        print(f"English: {en_translation}")
        
        # Translate English to Spanish
        en_text = "The artificial intelligence model performed well."
        es_translation = processor.translate_text(en_text, target_lang='es', source_lang='en')
        print(f"\nEnglish: {en_text}")
        print(f"Spanish: {es_translation}")
    else:
        print("⚠️ Translator not available for testing")
    
    print("\n\n📋 Supported Languages:")
    print("-" * 80)
    languages = processor.list_supported_languages()
    for lang in languages[:10]:  # Show first 10
        print(f"{lang['emoji']} {lang['code']}: {lang['name']}")
    print(f"... and {len(languages) - 10} more")
    
    print("\n✅ All tests completed!")