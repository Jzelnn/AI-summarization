"""
Multilingual Support Evaluation
Tests translation quality and language detection accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from nlp.multilingual import MultilingualProcessor
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    print("⚠️ Multilingual support not available")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from collections import defaultdict
import time

# Test dataset for language detection
LANGUAGE_DETECTION_DATASET = [
    {"text": "Artificial intelligence is transforming the world of technology.", "language": "en"},
    {"text": "This is another example of English text for testing purposes.", "language": "en"},
    {"text": "Machine learning models require large amounts of data.", "language": "en"},
    
    {"text": "Kecerdasan buatan mengubah dunia teknologi dengan cepat.", "language": "id"},
    {"text": "Sistem ini mendukung berbagai bahasa termasuk Bahasa Indonesia.", "language": "id"},
    {"text": "Teknologi pembelajaran mesin berkembang pesat di Indonesia.", "language": "id"},
    
    {"text": "La inteligencia artificial está transformando el mundo de la tecnología.", "language": "es"},
    {"text": "El aprendizaje automático requiere grandes cantidades de datos.", "language": "es"},
    {"text": "Los modelos de lenguaje natural son cada vez más precisos.", "language": "es"},
    
    {"text": "L'intelligence artificielle transforme le monde de la technologie.", "language": "fr"},
    {"text": "Les modèles d'apprentissage automatique nécessitent beaucoup de données.", "language": "fr"},
    {"text": "La reconnaissance d'entités nommées est une tâche importante.", "language": "fr"},
    
    {"text": "Künstliche Intelligenz verändert die Welt der Technologie.", "language": "de"},
    {"text": "Maschinelles Lernen erfordert große Datenmengen.", "language": "de"},
    {"text": "Sprachmodelle werden immer leistungsfähiger.", "language": "de"},
    
    {"text": "人工知能は技術の世界を変革しています。", "language": "ja"},
    {"text": "機械学習モデルは大量のデータを必要とします。", "language": "ja"},
    {"text": "自然言語処理の技術が進歩しています。", "language": "ja"},
    
    {"text": "人工智能正在改变技术世界。", "language": "zh-cn"},
    {"text": "机器学习模型需要大量数据。", "language": "zh-cn"},
    {"text": "自然语言处理技术不断进步。", "language": "zh-cn"},
    
    {"text": "인공지능은 기술 세계를 변화시키고 있습니다.", "language": "ko"},
    {"text": "기계 학습 모델은 많은 양의 데이터가 필요합니다.", "language": "ko"},
    {"text": "자연어 처리 기술이 발전하고 있습니다.", "language": "ko"},
    
    {"text": "الذكاء الاصطناعي يغير عالم التكنولوجيا.", "language": "ar"},
    {"text": "نماذج التعلم الآلي تتطلب كميات كبيرة من البيانات.", "language": "ar"},
    
    {"text": "Искусственный интеллект меняет мир технологий.", "language": "ru"},
    {"text": "Модели машинного обучения требуют больших объемов данных.", "language": "ru"},
    
    {"text": "A inteligência artificial está transformando o mundo da tecnologia.", "language": "pt"},
    {"text": "Modelos de aprendizado de máquina requerem grandes quantidades de dados.", "language": "pt"},
]

# Translation quality test pairs (source -> expected translation)
TRANSLATION_TEST_PAIRS = [
    {
        "source_text": "Lionel Messi mencetak gol untuk Argentina.",
        "source_lang": "id",
        "target_lang": "en",
        "reference_translation": "Lionel Messi scored a goal for Argentina.",
        "key_terms": ["Lionel Messi", "goal", "Argentina"]
    },
    {
        "source_text": "El presidente anunció nuevas políticas.",
        "source_lang": "es",
        "target_lang": "en",
        "reference_translation": "The president announced new policies.",
        "key_terms": ["president", "announced", "policies"]
    },
    {
        "source_text": "The artificial intelligence model is very accurate.",
        "source_lang": "en",
        "target_lang": "fr",
        "reference_translation": "Le modèle d'intelligence artificielle est très précis.",
        "key_terms": ["intelligence artificielle", "modèle", "précis"]
    },
    {
        "source_text": "Teknologi pembelajaran mesin berkembang pesat.",
        "source_lang": "id",
        "target_lang": "en",
        "reference_translation": "Machine learning technology is developing rapidly.",
        "key_terms": ["machine learning", "technology", "developing"]
    },
    {
        "source_text": "The team won the championship.",
        "source_lang": "en",
        "target_lang": "es",
        "reference_translation": "El equipo ganó el campeonato.",
        "key_terms": ["equipo", "ganó", "campeonato"]
    },
]


def evaluate_language_detection():
    """
    Evaluate language detection accuracy
    """
    if not MULTILINGUAL_AVAILABLE:
        print("⚠️ Multilingual support not available for testing")
        return None
    
    print("=" * 80)
    print("LANGUAGE DETECTION EVALUATION")
    print("=" * 80)
    
    processor = MultilingualProcessor()
    
    true_labels = []
    predicted_labels = []
    detection_times = []
    
    print(f"\n📊 Testing on {len(LANGUAGE_DETECTION_DATASET)} samples...\n")
    
    correct = 0
    total = 0
    
    for i, sample in enumerate(LANGUAGE_DETECTION_DATASET, 1):
        text = sample["text"]
        true_lang = sample["language"]
        
        start_time = time.time()
        predicted_lang = processor.detect_language(text)
        detection_time = time.time() - start_time
        
        detection_times.append(detection_time)
        
        # Normalize language codes
        predicted_lang_normalized = predicted_lang
        if predicted_lang == "zh-CN" or predicted_lang == "zh":
            predicted_lang_normalized = "zh-cn"
        
        true_lang_normalized = true_lang.lower()
        
        is_correct = predicted_lang_normalized == true_lang_normalized
        
        if is_correct:
            correct += 1
        
        true_labels.append(true_lang_normalized)
        predicted_labels.append(predicted_lang_normalized)
        
        total += 1
        
        # Show samples (first 5 and all errors)
        if i <= 5 or not is_correct:
            status = "✅" if is_correct else "❌"
            print(f"{status} Sample {i}:")
            print(f"   Text: {text[:70]}...")
            print(f"   True: {true_lang} | Predicted: {predicted_lang} | Time: {detection_time:.3f}s")
            if not is_correct:
                print(f"   ⚠️  MISDETECTION")
            print()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Get unique languages
    unique_languages = sorted(set(true_labels))
    
    # Per-language metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=unique_languages, average=None, zero_division=0
    )
    
    print("=" * 80)
    print("📊 LANGUAGE DETECTION RESULTS")
    print("=" * 80)
    print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
    print(f"⚡ Average Detection Time: {sum(detection_times)/len(detection_times):.3f}s")
    print(f"📝 Total Samples: {total}")
    print(f"✓ Correct: {correct}")
    print(f"✗ Incorrect: {total - correct}\n")
    
    print("=" * 80)
    print("📈 PER-LANGUAGE PERFORMANCE")
    print("=" * 80)
    print(f"\n{'Language':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, lang in enumerate(unique_languages):
        print(f"{lang.upper():<10} {precision[i]:<12.2%} {recall[i]:<12.2%} {f1[i]:<12.2%} {int(support[i]):<10}")
    
    results = {
        "overall_accuracy": float(accuracy),
        "average_detection_time": float(sum(detection_times)/len(detection_times)),
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "per_language": {
            unique_languages[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(len(unique_languages))
        }
    }
    
    return results


def calculate_translation_quality(translation, reference, key_terms):
    """
    Calculate translation quality metrics
    """
    translation_lower = translation.lower()
    reference_lower = reference.lower()
    
    # Key terms preservation
    terms_found = sum(1 for term in key_terms if term.lower() in translation_lower)
    term_preservation = terms_found / len(key_terms) if key_terms else 0
    
    # Simple word overlap (not perfect but gives indication)
    trans_words = set(translation_lower.split())
    ref_words = set(reference_lower.split())
    
    if len(trans_words) == 0:
        word_precision = 0
    else:
        word_precision = len(trans_words & ref_words) / len(trans_words)
    
    if len(ref_words) == 0:
        word_recall = 0
    else:
        word_recall = len(trans_words & ref_words) / len(ref_words)
    
    if word_precision + word_recall == 0:
        word_f1 = 0
    else:
        word_f1 = 2 * (word_precision * word_recall) / (word_precision + word_recall)
    
    # Length similarity
    length_ratio = min(len(translation), len(reference)) / max(len(translation), len(reference)) if max(len(translation), len(reference)) > 0 else 0
    
    return {
        "term_preservation": term_preservation,
        "word_precision": word_precision,
        "word_recall": word_recall,
        "word_f1": word_f1,
        "length_ratio": length_ratio
    }


def evaluate_translation_quality():
    """
    Evaluate translation quality
    """
    if not MULTILINGUAL_AVAILABLE:
        print("⚠️ Multilingual support not available for testing")
        return None
    
    print("\n" + "=" * 80)
    print("TRANSLATION QUALITY EVALUATION")
    print("=" * 80)
    
    processor = MultilingualProcessor()
    
    all_metrics = []
    translation_times = []
    
    print(f"\n📊 Testing on {len(TRANSLATION_TEST_PAIRS)} translation pairs...\n")
    
    for i, pair in enumerate(TRANSLATION_TEST_PAIRS, 1):
        source_text = pair["source_text"]
        source_lang = pair["source_lang"]
        target_lang = pair["target_lang"]
        reference = pair["reference_translation"]
        key_terms = pair["key_terms"]
        
        print(f"{'='*80}")
        print(f"Translation Pair {i}:")
        print(f"Source ({source_lang}): {source_text}")
        print(f"Reference ({target_lang}): {reference}")
        
        start_time = time.time()
        translation = processor.translate_text(source_text, target_lang, source_lang)
        translation_time = time.time() - start_time
        translation_times.append(translation_time)
        
        print(f"Predicted ({target_lang}): {translation}")
        print(f"Time: {translation_time:.2f}s")
        
        # Calculate quality metrics
        metrics = calculate_translation_quality(translation, reference, key_terms)
        all_metrics.append(metrics)
        
        print(f"\n📊 Quality Metrics:")
        print(f"   • Key Terms Preserved: {metrics['term_preservation']:.2%}")
        print(f"   • Word Precision: {metrics['word_precision']:.2%}")
        print(f"   • Word Recall: {metrics['word_recall']:.2%}")
        print(f"   • Word F1-Score: {metrics['word_f1']:.2%}")
        print(f"   • Length Similarity: {metrics['length_ratio']:.2%}")
        print()
    
    # Average metrics
    avg_metrics = {
        "average_term_preservation": sum(m["term_preservation"] for m in all_metrics) / len(all_metrics),
        "average_word_precision": sum(m["word_precision"] for m in all_metrics) / len(all_metrics),
        "average_word_recall": sum(m["word_recall"] for m in all_metrics) / len(all_metrics),
        "average_word_f1": sum(m["word_f1"] for m in all_metrics) / len(all_metrics),
        "average_length_ratio": sum(m["length_ratio"] for m in all_metrics) / len(all_metrics),
        "average_translation_time": sum(translation_times) / len(translation_times)
    }
    
    print("=" * 80)
    print("📊 AVERAGE TRANSLATION QUALITY")
    print("=" * 80)
    print(f"\n✅ Key Terms Preservation: {avg_metrics['average_term_preservation']:.2%}")
    print(f"📝 Word Precision: {avg_metrics['average_word_precision']:.2%}")
    print(f"🎯 Word Recall: {avg_metrics['average_word_recall']:.2%}")
    print(f"⭐ Word F1-Score: {avg_metrics['average_word_f1']:.2%}")
    print(f"📏 Length Similarity: {avg_metrics['average_length_ratio']:.2%}")
    print(f"⚡ Average Translation Time: {avg_metrics['average_translation_time']:.2f}s")
    
    results = {
        "average_metrics": avg_metrics,
        "individual_translations": [
            {
                "source": pair["source_text"],
                "source_lang": pair["source_lang"],
                "target_lang": pair["target_lang"],
                "reference": pair["reference_translation"],
                "metrics": all_metrics[i]
            }
            for i, pair in enumerate(TRANSLATION_TEST_PAIRS)
        ]
    }
    
    return results


def evaluate_multilingual_support():
    """
    Main evaluation function
    """
    print("=" * 80)
    print("MULTILINGUAL SUPPORT COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    if not MULTILINGUAL_AVAILABLE:
        print("\n⚠️  Multilingual support not available")
        print("Please install required packages:")
        print("  pip install deep-translator langdetect")
        return
    
    # Test 1: Language Detection
    detection_results = evaluate_language_detection()
    
    # Test 2: Translation Quality
    translation_results = evaluate_translation_quality()
    
    # Combine results
    final_results = {
        "language_detection": detection_results,
        "translation_quality": translation_results,
        "summary": {
            "detection_accuracy": detection_results["overall_accuracy"] if detection_results else 0,
            "translation_quality_f1": translation_results["average_metrics"]["average_word_f1"] if translation_results else 0,
            "key_terms_preservation": translation_results["average_metrics"]["average_term_preservation"] if translation_results else 0
        }
    }
    
    # Save results
    output_path = "/mnt/user-data/outputs/multilingual_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Results saved to: {output_path}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("✅ MULTILINGUAL EVALUATION COMPLETE!")
    print("=" * 80)
    
    if detection_results and translation_results:
        print(f"\n🎯 Key Findings:")
        print(f"   • Language Detection Accuracy: {detection_results['overall_accuracy']:.2%}")
        print(f"   • Average Translation Quality (F1): {translation_results['average_metrics']['average_word_f1']:.2%}")
        print(f"   • Key Terms Preservation: {translation_results['average_metrics']['average_term_preservation']:.2%}")
        print(f"   • Languages Tested: {len(detection_results['per_language'])}")
    
    return final_results


if __name__ == "__main__":
    evaluate_multilingual_support()