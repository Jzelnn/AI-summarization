"""
Evaluation Metrics for Summarization Models
Supports ROUGE, BLEU, F1-Score, and other quality metrics
"""

from typing import Dict, List, Tuple
import re
from collections import Counter
import numpy as np

# Try to import advanced metrics libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️ NLTK not installed. Install with: pip install nltk")


class SummarizationEvaluator:
    """Comprehensive evaluation metrics for summarization tasks"""
    
    def __init__(self):
        """Initialize the evaluator with available metrics"""
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
    
    def evaluate(
        self,
        reference: str,
        generated: str,
        include_all_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated summary against reference
        
        Args:
            reference: Reference/ground truth summary or original text
            generated: Generated summary to evaluate
            include_all_metrics: Whether to include all available metrics
            
        Returns:
            Dictionary containing all evaluation scores
        """
        metrics = {}
        
        # Basic metrics (always available)
        metrics.update(self._basic_metrics(reference, generated))
        
        if include_all_metrics:
            # ROUGE scores (if available)
            if self.rouge_scorer:
                metrics.update(self._rouge_scores(reference, generated))
            
            # BLEU score (if available)
            if BLEU_AVAILABLE:
                metrics['bleu'] = self._bleu_score(reference, generated)
            
            # Content preservation metrics
            metrics.update(self._content_metrics(reference, generated))
            
            # Readability metrics
            metrics.update(self._readability_metrics(generated))
        
        return metrics
    
    def _basic_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        """Calculate basic length and compression metrics"""
        ref_words = reference.split()
        gen_words = generated.split()
        
        return {
            'compression_ratio': len(generated) / len(reference) if len(reference) > 0 else 0,
            'word_count_ratio': len(gen_words) / len(ref_words) if len(ref_words) > 0 else 0,
            'summary_length': len(generated),
            'summary_words': len(gen_words),
            'original_length': len(reference),
            'original_words': len(ref_words)
        }
    
    def _rouge_scores(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        
        ROUGE-1: Unigram overlap
        ROUGE-2: Bigram overlap  
        ROUGE-L: Longest common subsequence
        """
        if not self.rouge_scorer:
            return {}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
    
    def _bleu_score(self, reference: str, generated: str) -> float:
        """
        Calculate BLEU score (0-1)
        Higher is better, measures n-gram precision
        """
        if not BLEU_AVAILABLE:
            return 0.0
        
        try:
            ref_tokens = word_tokenize(reference.lower())
            gen_tokens = word_tokenize(generated.lower())
            
            # Use smoothing for short sequences
            smoothie = SmoothingFunction().method4
            score = sentence_bleu(
                [ref_tokens],
                gen_tokens,
                smoothing_function=smoothie
            )
            
            return score
        except Exception as e:
            print(f"⚠️ BLEU calculation error: {e}")
            return 0.0
    
    def _content_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Measure content preservation and overlap
        
        Returns:
            - content_f1: F1 score of word overlap
            - unique_words_preserved: % of unique words preserved
            - named_entity_preservation: Approximate NE preservation
        """
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        
        # Calculate F1 score
        if len(gen_words) == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            overlap = len(ref_words & gen_words)
            precision = overlap / len(gen_words) if len(gen_words) > 0 else 0
            recall = overlap / len(ref_words) if len(ref_words) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Approximate named entity preservation (capitalized words)
        ref_entities = set(word for word in reference.split() if word and word[0].isupper())
        gen_entities = set(word for word in generated.split() if word and word[0].isupper())
        
        entity_preservation = (
            len(ref_entities & gen_entities) / len(ref_entities) 
            if len(ref_entities) > 0 else 1.0
        )
        
        return {
            'content_precision': precision,
            'content_recall': recall,
            'content_f1': f1,
            'unique_words_preserved': len(ref_words & gen_words) / len(ref_words) if len(ref_words) > 0 else 0,
            'entity_preservation': entity_preservation
        }
    
    def _readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Simple readability metrics
        
        Returns:
            - avg_sentence_length: Average words per sentence
            - avg_word_length: Average characters per word
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        
        avg_sent_len = len(words) / len(sentences) if len(sentences) > 0 else 0
        avg_word_len = sum(len(w) for w in words) / len(words) if len(words) > 0 else 0
        
        return {
            'avg_sentence_length': avg_sent_len,
            'avg_word_length': avg_word_len,
            'sentence_count': len(sentences)
        }
    
    def compare_models(
        self,
        reference: str,
        summaries: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple model outputs
        
        Args:
            reference: Original text or reference summary
            summaries: Dict of {model_name: summary_text}
            
        Returns:
            Dict of {model_name: metrics_dict}
        """
        results = {}
        
        for model_name, summary in summaries.items():
            print(f"📊 Evaluating {model_name}...")
            results[model_name] = self.evaluate(reference, summary)
        
        return results
    
    def get_best_model(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metric: str = 'rouge1_f1'
    ) -> Tuple[str, float]:
        """
        Determine best model based on a specific metric
        
        Args:
            comparison_results: Results from compare_models()
            metric: Metric to use for comparison
            
        Returns:
            (best_model_name, score)
        """
        best_model = None
        best_score = -1
        
        for model_name, metrics in comparison_results.items():
            score = metrics.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score
    
    def format_results(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display"""
        lines = ["📊 Evaluation Metrics:", "=" * 60]
        
        # Group metrics by category
        categories = {
            "Length Metrics": ['compression_ratio', 'word_count_ratio', 'summary_words', 'original_words'],
            "ROUGE Scores": ['rouge1_f1', 'rouge2_f1', 'rougeL_f1'],
            "Content Preservation": ['content_f1', 'unique_words_preserved', 'entity_preservation'],
            "Quality Metrics": ['bleu', 'avg_sentence_length', 'avg_word_length']
        }
        
        for category, metric_names in categories.items():
            category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
            if category_metrics:
                lines.append(f"\n{category}:")
                for name, value in category_metrics.items():
                    if isinstance(value, float):
                        if name.endswith('_ratio') or name.endswith('_f1') or 'preserved' in name:
                            lines.append(f"  • {name}: {value:.2%}")
                        else:
                            lines.append(f"  • {name}: {value:.2f}")
                    else:
                        lines.append(f"  • {name}: {value}")
        
        return "\n".join(lines)


# Simple fallback implementations if libraries aren't available
def simple_rouge1(reference: str, generated: str) -> float:
    """Simple ROUGE-1 F1 calculation without external library"""
    ref_words = set(reference.lower().split())
    gen_words = set(generated.lower().split())
    
    if len(gen_words) == 0:
        return 0.0
    
    overlap = len(ref_words & gen_words)
    precision = overlap / len(gen_words)
    recall = overlap / len(ref_words) if len(ref_words) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def simple_content_score(reference: str, generated: str) -> Dict[str, float]:
    """Simple content preservation score"""
    ref_words = reference.lower().split()
    gen_words = generated.lower().split()
    
    ref_set = set(ref_words)
    gen_set = set(gen_words)
    
    overlap = len(ref_set & gen_set)
    
    precision = overlap / len(gen_set) if len(gen_set) > 0 else 0
    recall = overlap / len(ref_set) if len(ref_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'content_precision': precision,
        'content_recall': recall,
        'content_f1': f1,
        'compression_ratio': len(generated) / len(reference) if len(reference) > 0 else 0
    }


# Testing
if __name__ == "__main__":
    # Test data
    original = """
    Lionel Messi scored a stunning goal last night as Argentina defeated Brazil 2-1 
    in a thrilling Copa America final. The match was held in Buenos Aires and watched 
    by over 80,000 fans at the stadium. Messi once again proved why he is considered 
    one of the greatest players of all time.
    """
    
    summary1 = "Messi scored as Argentina beat Brazil 2-1 in Copa America final in Buenos Aires."
    summary2 = "Argentina won the Copa America final against Brazil with a 2-1 victory."
    
    print("=" * 80)
    print("TESTING EVALUATION METRICS")
    print("=" * 80)
    
    evaluator = SummarizationEvaluator()
    
    # Single model evaluation
    print("\n📊 Evaluating Summary 1:")
    print("-" * 80)
    metrics1 = evaluator.evaluate(original, summary1)
    print(evaluator.format_results(metrics1))
    
    # Model comparison
    print("\n\n" + "=" * 80)
    print("🔍 COMPARING MODELS")
    print("=" * 80)
    
    comparison = evaluator.compare_models(
        original,
        {
            'Gemini': summary1,
            'T5-Small': summary2
        }
    )
    
    for model, metrics in comparison.items():
        print(f"\n{model} Performance:")
        print("-" * 80)
        print(evaluator.format_results(metrics))
    
    # Find best model
    best_model, best_score = evaluator.get_best_model(comparison, 'content_f1')
    print(f"\n\n🏆 Best Model: {best_model} (F1 Score: {best_score:.2%})")