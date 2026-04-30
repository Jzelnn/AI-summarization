"""
AI-Powered Summarizer using Google Gemini
Free tier: 15 requests per minute, 1500 requests per day
"""

import google.generativeai as genai
from typing import Literal
import os
import re
from functools import lru_cache

class GeminiSummarizer:
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash-latest"):
        """
        Initialize Gemini summarizer
        
        Args:
            api_key: Google Gemini API key (get from https://makersuite.google.com/app/apikey)
                    If None, will try to read from GEMINI_API_KEY environment variable
            model_name: Model to use. Options:
                - "gemini-1.5-flash-latest" (default, fastest, free)
                - "gemini-1.5-pro-latest" (more capable, slower)
                - "gemini-pro" (older, still good)
        """
        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Get one from: https://makersuite.google.com/app/apikey\n"
                "Set it as environment variable: $env:GEMINI_API_KEY='your-key' (PowerShell)\n"
                "Or pass it directly: GeminiSummarizer(api_key='your-key')"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Try to initialize the model
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.model_name = model_name
            print(f"✅ Gemini AI initialized successfully with model: {model_name}")
        except Exception as e:
            print(f"⚠️ Error initializing {model_name}: {e}")
            print("Trying fallback model: gemini-pro")
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.model_name = 'gemini-pro'
                print("✅ Gemini AI initialized with fallback model: gemini-pro")
            except Exception as e2:
                raise ValueError(f"Failed to initialize any Gemini model. Error: {e2}")
    
    def _create_prompt(
        self,
        text: str,
        level: str,
        topic: str | None = None
    ) -> str:
        """
        Create optimized prompt for Gemini
        
        Args:
            text: Input text
            level: Summary level
            topic: Optional topic
            
        Returns:
            Formatted prompt
        """
        # Base instructions
        if level == "short":
            instruction = """Create a VERY CONCISE summary in bullet points (•).
            - Use 3-5 bullet points maximum
            - Each bullet should be one clear, short sentence
            - Focus only on the most critical information
            - Format: • Point 1\\n• Point 2\\n• Point 3"""
        
        elif level == "long":
            instruction = """Create a DETAILED summary that:
            - Covers all major points and supporting details
            - Maintains the narrative flow and context
            - Includes relevant examples and specific information
            - Is about 40-50% of the original length
            - Written in clear, flowing paragraphs"""
        
        else:  # medium
            instruction = """Create a BALANCED summary that:
            - Captures the main ideas and key supporting points
            - Is concise but complete
            - Is about 25-30% of the original length
            - Written in clear, coherent sentences"""
        
        # Add topic context
        topic_context = ""
        if topic and topic != "general":
            topic_context = f"\n\nContext: This text is about {topic}. Pay special attention to {topic}-related terms and concepts."
        
        # Construct full prompt
        prompt = f"""{instruction}{topic_context}

Text to summarize:
{text}

Summary:"""
        
        return prompt
    
    def summarize(
        self,
        text: str,
        level: Literal["short", "medium", "long"] = "medium",
        topic: str | None = None,
        temperature: float = 0.3
    ) -> str:
        """
        Generate AI-powered summary using Gemini
        
        Args:
            text: Input text to summarize
            level: Summary length - "short", "medium", or "long"
            topic: Optional topic for context (sports, politics, tech, general)
            temperature: Creativity level (0.0-1.0, lower = more focused)
            
        Returns:
            AI-generated summary
        """
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, level, topic)
            
            # Generate summary
            print(f"🤖 Generating {level} summary with Gemini...")
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=1024,
                )
            )
            
            summary = response.text.strip()
            
            print(f"✅ Summary generated: {len(summary.split())} words")
            return summary
            
        except Exception as e:
            print(f"❌ Error with Gemini API: {e}")
            # Fallback to extractive summarization
            return self._extractive_fallback(text, level)
    
    def summarize_with_analysis(
        self,
        text: str,
        level: Literal["short", "medium", "long"] = "medium",
        topic: str | None = None
    ) -> dict:
        """
        Generate summary with additional analysis
        
        Args:
            text: Input text
            level: Summary level
            topic: Optional topic
            
        Returns:
            Dictionary with summary, sentiment, and key themes
        """
        # Get summary
        summary = self.summarize(text, level, topic)
        
        # Get additional analysis
        analysis_prompt = f"""Analyze this text and provide:
        1. Overall sentiment (positive/negative/neutral)
        2. Top 3 key themes or topics
        3. Main entities mentioned (people, organizations, locations)

        Text:
        {text}

        Respond in this exact format:
        SENTIMENT: [sentiment]
        THEMES: [theme1], [theme2], [theme3]
        ENTITIES: [entity1], [entity2], [entity3]
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            analysis_text = response.text.strip()
            
            # Parse response
            analysis = {
                "summary": summary,
                "sentiment": "neutral",
                "themes": [],
                "key_entities": []
            }
            
            for line in analysis_text.split('\n'):
                if line.startswith("SENTIMENT:"):
                    analysis["sentiment"] = line.split(":", 1)[1].strip()
                elif line.startswith("THEMES:"):
                    themes_text = line.split(":", 1)[1].strip()
                    analysis["themes"] = [t.strip() for t in themes_text.split(",")]
                elif line.startswith("ENTITIES:"):
                    entities_text = line.split(":", 1)[1].strip()
                    analysis["key_entities"] = [e.strip() for e in entities_text.split(",")]
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ Analysis failed: {e}")
            return {"summary": summary}
    
    def batch_summarize(
        self,
        texts: list[str],
        level: Literal["short", "medium", "long"] = "medium",
        topic: str | None = None
    ) -> list[str]:
        """
        Summarize multiple texts efficiently
        
        Args:
            texts: List of texts to summarize
            level: Summary level
            topic: Optional topic
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i, text in enumerate(texts, 1):
            print(f"Processing {i}/{len(texts)}...")
            summary = self.summarize(text, level, topic)
            summaries.append(summary)
        
        return summaries
    
    def _extractive_fallback(self, text: str, level: str) -> str:
        """
        Fallback extractive summarization if API fails
        
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


# Singleton instance
_gemini_instance = None

def get_gemini_summarizer(api_key: str | None = None) -> GeminiSummarizer:
    """
    Get or create Gemini summarizer instance
    
    Args:
        api_key: Optional API key
        
    Returns:
        GeminiSummarizer instance
    """
    global _gemini_instance
    
    if _gemini_instance is None:
        _gemini_instance = GeminiSummarizer(api_key)
    
    return _gemini_instance


def summarize(
    text: str,
    level: Literal["short", "medium", "long"] = "medium",
    topic: str | None = None,
    api_key: str | None = None
) -> str:
    """
    Main summarization function (drop-in replacement)
    
    Args:
        text: Input text
        level: Summary level
        topic: Optional topic
        api_key: Optional Gemini API key
        
    Returns:
        AI-generated summary
    """
    summarizer = get_gemini_summarizer(api_key)
    return summarizer.summarize(text, level, topic)


# Test the summarizer
if __name__ == "__main__":
    # You need to set your API key first
    # Get it from: https://makersuite.google.com/app/apikey
    
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
    print("TESTING GEMINI SUMMARIZER")
    print("="*80)
    
    try:
        for level in ["short", "medium", "long"]:
            print(f"\n🔹 Level: {level.upper()}")
            summary = summarize(test_text, level=level, topic="sports")
            print(f"Summary:\n{summary}")
            print("-"*80)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nTo test Gemini:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GEMINI_API_KEY='your-key'")
        print("3. Or pass directly: summarize(text, api_key='your-key')")