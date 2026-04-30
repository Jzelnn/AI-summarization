"""
AI-Powered Summarizer using HuggingFace Transformers
Supports multiple summarization levels with topic-aware processing
FIXED: httpx.TimeoutException compatibility issue
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Literal
import re

class AISummarizer:
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the AI summarizer
        
        Args:
            model_name: HuggingFace model to use
                - "t5-small" (default, fast and lightweight ~60MB)
                - "facebook/bart-large-cnn" (better quality, larger ~1.6GB)
                - "google/pegasus-xsum" (shorter summaries)
                - "philschmid/bart-large-cnn-samsum" (good for conversations)
        """
        print(f"🤖 Loading AI model: {model_name}...")
        
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            # FIXED: Load model and tokenizer explicitly to avoid httpx issues
            if model_name == "t5-small":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Move model to device
                if device == 0:
                    self.model = self.model.cuda()
                
                # Create pipeline with explicit model and tokenizer
                self.summarizer = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    framework="pt"
                )
            else:
                # For other models, use standard pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    device=device
                )
            
            self.model_name = model_name
            print(f"✅ Model loaded successfully on {'GPU' if device == 0 else 'CPU'}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to default summarization...")
            raise
    
    def _chunk_text(self, text: str, max_length: int = 512) -> list[str]:
        """
        Split long text into chunks that the model can process
        
        Args:
            text: Input text
            max_length: Maximum tokens per chunk (T5-small limit is 512)
            
        Returns:
            List of text chunks
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Single sentence is too long, split it
                    words = sentence.split()
                    chunks.append(' '.join(words[:max_length]))
                    current_chunk = words[max_length:]
                    current_length = len(current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_length_params(self, level: str, text_length: int) -> dict:
        """
        Get min/max length parameters based on summary level
        
        Args:
            level: "short", "medium", or "long"
            text_length: Length of input text in words
            
        Returns:
            Dictionary with min_length and max_length
        """
        if level == "short":
            # Bullet points style - very concise
            return {
                "min_length": max(10, text_length // 20),
                "max_length": max(50, text_length // 10),
            }
        elif level == "long":
            # Detailed summary
            return {
                "min_length": max(50, text_length // 6),
                "max_length": max(200, text_length // 3),
            }
        else:  # medium
            # Standard summary
            return {
                "min_length": max(30, text_length // 10),
                "max_length": max(130, text_length // 5),
            }
    
    def summarize(
        self,
        text: str,
        level: Literal["short", "medium", "long"] = "medium",
        topic: str | None = None
    ) -> str:
        """
        Generate AI-powered summary
        
        Args:
            text: Input text to summarize
            level: Summary length - "short", "medium", or "long"
            topic: Optional topic for context (sports, politics, tech, general)
            
        Returns:
            AI-generated summary
        """
        if not text or not text.strip():
            return ""
        
        # Clean the text
        text = text.strip()
        text_length = len(text.split())
        
        # Get length parameters
        params = self._get_length_params(level, text_length)
        
        # FIXED: For T5, we need to add "summarize: " prefix
        if self.model_name == "t5-small":
            # Add topic context if provided
            if topic and topic != "general":
                prefix = f"summarize: This is about {topic}. "
            else:
                prefix = "summarize: "
            text_to_summarize = prefix + text
        else:
            # For other models
            if topic and topic != "general":
                prefix = f"This is about {topic}. "
                text_to_summarize = prefix + text
            else:
                text_to_summarize = text
        
        try:
            # For very long texts, process in chunks
            if text_length > 400:  # Reduced from 1024 to 400 for T5-small
                print(f"📄 Processing long text ({text_length} words) in chunks...")
                chunks = self._chunk_text(text, max_length=400)
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    print(f"   Processing chunk {i+1}/{len(chunks)}...")
                    
                    chunk_to_process = f"summarize: {chunk}" if self.model_name == "t5-small" else chunk
                    
                    result = self.summarizer(
                        chunk_to_process,
                        min_length=params["min_length"] // len(chunks),
                        max_length=params["max_length"] // len(chunks),
                        do_sample=False,
                        truncation=True
                    )
                    
                    # Extract text from result
                    if isinstance(result, list) and len(result) > 0:
                        summary_text = result[0].get('generated_text', '') or result[0].get('summary_text', '')
                        summaries.append(summary_text)
                
                # Combine chunk summaries
                combined = " ".join(summaries)
                
                # If needed, summarize the summaries
                if len(combined.split()) > params["max_length"]:
                    combined_to_process = f"summarize: {combined}" if self.model_name == "t5-small" else combined
                    
                    final_result = self.summarizer(
                        combined_to_process,
                        min_length=params["min_length"],
                        max_length=params["max_length"],
                        do_sample=False,
                        truncation=True
                    )
                    
                    if isinstance(final_result, list) and len(final_result) > 0:
                        summary = final_result[0].get('generated_text', '') or final_result[0].get('summary_text', '')
                    else:
                        summary = combined
                else:
                    summary = combined
            else:
                # Process normally for shorter texts
                result = self.summarizer(
                    text_to_summarize,
                    min_length=params["min_length"],
                    max_length=params["max_length"],
                    do_sample=False,
                    truncation=True
                )
                
                # Extract summary from result
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get('generated_text', '') or result[0].get('summary_text', '')
                else:
                    summary = str(result)
            
            # For short level, convert to bullet points
            if level == "short":
                summary = self._convert_to_bullets(summary)
            
            print(f"✅ Summary generated: {len(summary.split())} words")
            return summary.strip()
            
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            print(f"   Full error: {type(e).__name__}")
            # Fallback to extractive summarization
            return self._extractive_fallback(text, level)
    
    def _convert_to_bullets(self, summary: str) -> str:
        """
        Convert summary to bullet points
        
        Args:
            summary: Summary text
            
        Returns:
            Bullet-pointed summary
        """
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        bullets = []
        
        for sentence in sentences:
            if sentence.strip():
                # Clean and format
                sentence = sentence.strip()
                if not sentence.endswith('.'):
                    sentence += '.'
                bullets.append(f"• {sentence}")
        
        return "\n".join(bullets)
    
    def _extractive_fallback(self, text: str, level: str) -> str:
        """
        Fallback extractive summarization if AI fails
        
        Args:
            text: Input text
            level: Summary level
            
        Returns:
            Extractive summary
        """
        from collections import Counter
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        words = re.findall(r'\w+', text.lower())
        freq = Counter(words)
        
        # Score sentences
        scores = {}
        for sent in sentences:
            score = sum(freq.get(word.lower(), 0) for word in sent.split())
            scores[sent] = score
        
        # Select top sentences
        if level == "short":
            n = max(1, len(sentences) // 5)
            # Convert to bullets
            ranked = sorted(scores, key=scores.get, reverse=True)[:n]
            return "\n".join(f"• {s}" for s in ranked)
        elif level == "long":
            n = max(1, len(sentences) // 2)
        else:
            n = max(1, len(sentences) // 3)
        
        ranked = sorted(scores, key=scores.get, reverse=True)
        return " ".join(ranked[:n])


# Singleton instance for reuse
_summarizer_instance = None

def get_summarizer(model_name: str = "t5-small") -> AISummarizer:
    """
    Get or create summarizer instance
    
    Args:
        model_name: HuggingFace model to use
        
    Returns:
        AISummarizer instance
    """
    global _summarizer_instance
    
    if _summarizer_instance is None or _summarizer_instance.model_name != model_name:
        _summarizer_instance = AISummarizer(model_name)
    
    return _summarizer_instance


def summarize(
    text: str,
    level: Literal["short", "medium", "long"] = "medium",
    topic: str | None = None,
    model_name: str = "t5-small"
) -> str:
    """
    Main summarization function (drop-in replacement for old summarizer)
    
    Args:
        text: Input text
        level: Summary level - "short", "medium", or "long"
        topic: Optional topic for context
        model_name: HuggingFace model to use
        
    Returns:
        AI-generated summary
    """
    summarizer = get_summarizer(model_name)
    return summarizer.summarize(text, level, topic)


# Test the summarizer
if __name__ == "__main__":
    test_text = """
    Lionel Messi scored a stunning goal last night as Argentina defeated Brazil 2-1 
    in a thrilling Copa America final. The match was held in Buenos Aires and watched 
    by over 80,000 fans at the stadium. Messi once again proved why he is considered 
    one of the greatest players of all time, displaying exceptional skill and leadership. 
    The Argentine captain opened the scoring in the 23rd minute with a brilliant free kick, 
    before Brazil equalized through Neymar. However, Messi set up the winning goal in 
    injury time, confirming Argentina's dominance in South American football.
    """
    
    print("="*80)
    print("TESTING AI SUMMARIZER (HuggingFace) - FIXED VERSION")
    print("="*80)
    
    try:
        for level in ["short", "medium", "long"]:
            print(f"\n🔹 Level: {level.upper()}")
            summary = summarize(test_text, level=level, topic="sports")
            print(f"Summary:\n{summary}")
            print("-"*80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()