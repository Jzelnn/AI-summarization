"""
Named Entity Recognition (NER) Performance Evaluation
Tests spaCy NER model with precision, recall, F1-score
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp.ner import extract_entities
from sklearn.metrics import precision_recall_fscore_support, classification_report
import json
from collections import defaultdict

# Test dataset with ground truth entities
NER_TEST_DATASET = [
    # English texts with entities
    {
        "text": "Lionel Messi scored a goal for Argentina at the World Cup in Qatar.",
        "entities": {
            "PERSON": ["Lionel Messi"],
            "GPE": ["Argentina", "Qatar"],
            "EVENT": ["World Cup"]
        },
        "language": "English"
    },
    {
        "text": "Apple Inc. CEO Tim Cook announced new products in California.",
        "entities": {
            "ORG": ["Apple Inc."],
            "PERSON": ["Tim Cook"],
            "GPE": ["California"]
        },
        "language": "English"
    },
    {
        "text": "The United Nations held a meeting in New York City on January 15, 2024.",
        "entities": {
            "ORG": ["United Nations"],
            "GPE": ["New York City"],
            "DATE": ["January 15, 2024"]
        },
        "language": "English"
    },
    {
        "text": "Manchester United defeated Liverpool 3-1 in the Premier League.",
        "entities": {
            "ORG": ["Manchester United", "Liverpool", "Premier League"]
        },
        "language": "English"
    },
    {
        "text": "Cristiano Ronaldo joined Al Nassr in Saudi Arabia.",
        "entities": {
            "PERSON": ["Cristiano Ronaldo"],
            "ORG": ["Al Nassr"],
            "GPE": ["Saudi Arabia"]
        },
        "language": "English"
    },
    {
        "text": "President Joe Biden met with Prime Minister Rishi Sunak in London.",
        "entities": {
            "PERSON": ["Joe Biden", "Rishi Sunak"],
            "GPE": ["London"]
        },
        "language": "English"
    },
    {
        "text": "Google announced a new AI model called Gemini at their headquarters in Mountain View.",
        "entities": {
            "ORG": ["Google"],
            "PRODUCT": ["Gemini"],
            "GPE": ["Mountain View"]
        },
        "language": "English"
    },
    {
        "text": "Elon Musk founded SpaceX and Tesla Motors in the United States.",
        "entities": {
            "PERSON": ["Elon Musk"],
            "ORG": ["SpaceX", "Tesla Motors"],
            "GPE": ["United States"]
        },
        "language": "English"
    },
    {
        "text": "The European Union imposed sanctions on Russia after the invasion of Ukraine.",
        "entities": {
            "ORG": ["European Union"],
            "GPE": ["Russia", "Ukraine"]
        },
        "language": "English"
    },
    {
        "text": "Amazon founder Jeff Bezos launched Blue Origin in Seattle, Washington.",
        "entities": {
            "ORG": ["Amazon", "Blue Origin"],
            "PERSON": ["Jeff Bezos"],
            "GPE": ["Seattle", "Washington"]
        },
        "language": "English"
    },
    {
        "text": "The Supreme Court ruled on the case in Washington D.C. on Monday.",
        "entities": {
            "ORG": ["Supreme Court"],
            "GPE": ["Washington D.C."],
            "DATE": ["Monday"]
        },
        "language": "English"
    },
    {
        "text": "Neymar plays for Brazil national team and Paris Saint-Germain.",
        "entities": {
            "PERSON": ["Neymar"],
            "ORG": ["Brazil national team", "Paris Saint-Germain"]
        },
        "language": "English"
    },
    {
        "text": "Microsoft acquired Activision Blizzard for $69 billion in 2023.",
        "entities": {
            "ORG": ["Microsoft", "Activision Blizzard"],
            "MONEY": ["$69 billion"],
            "DATE": ["2023"]
        },
        "language": "English"
    },
    {
        "text": "Taylor Swift performed at MetLife Stadium in New Jersey.",
        "entities": {
            "PERSON": ["Taylor Swift"],
            "FAC": ["MetLife Stadium"],
            "GPE": ["New Jersey"]
        },
        "language": "English"
    },
    {
        "text": "The Olympics will be held in Paris, France in July 2024.",
        "entities": {
            "EVENT": ["Olympics"],
            "GPE": ["Paris", "France"],
            "DATE": ["July 2024"]
        },
        "language": "English"
    },
]


def normalize_entity_type(entity_type):
    """
    Normalize entity types for comparison
    Some models use different labels for similar entities
    """
    mapping = {
        "TEAM": "ORG",  # Custom TEAM entities map to ORG
        "PLAYER": "PERSON",  # Custom PLAYER entities map to PERSON
        "GPE": "GPE",  # Geo-political entity (country, city, state)
        "LOC": "GPE",  # Location can be treated as GPE
        "NORP": "ORG",  # Nationalities/Groups can map to ORG
    }
    return mapping.get(entity_type, entity_type)


def evaluate_ner():
    """
    Evaluate NER performance with precision, recall, F1-score
    """
    print("=" * 80)
    print("NAMED ENTITY RECOGNITION (NER) PERFORMANCE EVALUATION")
    print("=" * 80)
    
    # Track predictions per entity type
    entity_types = set()
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    
    # Detailed results for analysis
    detailed_results = []
    
    print(f"\n📊 Testing on {len(NER_TEST_DATASET)} documents...\n")
    
    for i, sample in enumerate(NER_TEST_DATASET, 1):
        text = sample["text"]
        true_entities = sample["entities"]
        language = sample["language"]
        
        # Get predictions
        predicted_entities = extract_entities(text, use_filtering=True)
        
        # Show sample
        print(f"{'='*80}")
        print(f"Sample {i} ({language}):")
        print(f"Text: {text}")
        print(f"\n✓ True Entities:")
        for entity_type, items in true_entities.items():
            entity_types.add(entity_type)
            print(f"   {entity_type}: {items}")
        
        print(f"\n🔍 Predicted Entities:")
        for entity_type, items in predicted_entities.items():
            entity_types.add(normalize_entity_type(entity_type))
            print(f"   {entity_type}: {items}")
        
        # Calculate matches
        sample_result = {
            "text": text,
            "language": language,
            "true_entities": true_entities,
            "predicted_entities": predicted_entities,
            "matches": {},
            "missed": {},
            "incorrect": {}
        }
        
        # For each entity type
        all_entity_types = set(true_entities.keys()) | set(predicted_entities.keys())
        
        for entity_type in all_entity_types:
            true_set = set(true_entities.get(entity_type, []))
            pred_set = set(predicted_entities.get(entity_type, []))
            
            # Also check normalized versions
            normalized_type = normalize_entity_type(entity_type)
            for pred_type, pred_items in predicted_entities.items():
                if normalize_entity_type(pred_type) == normalized_type:
                    pred_set.update(pred_items)
            
            # Calculate TP, FP, FN
            matches = true_set & pred_set
            missed = true_set - pred_set
            incorrect = pred_set - true_set
            
            true_positives[entity_type] += len(matches)
            false_negatives[entity_type] += len(missed)
            false_positives[entity_type] += len(incorrect)
            
            sample_result["matches"][entity_type] = list(matches)
            sample_result["missed"][entity_type] = list(missed)
            sample_result["incorrect"][entity_type] = list(incorrect)
            
            # Print analysis
            if matches:
                print(f"\n   ✅ {entity_type} - Correct matches: {list(matches)}")
            if missed:
                print(f"   ❌ {entity_type} - Missed: {list(missed)}")
            if incorrect:
                print(f"   ⚠️  {entity_type} - Incorrect: {list(incorrect)}")
        
        detailed_results.append(sample_result)
        print()
    
    # Calculate metrics per entity type
    print("=" * 80)
    print("📈 PER-ENTITY-TYPE PERFORMANCE")
    print("=" * 80)
    print(f"\n{'Entity Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 95)
    
    metrics_by_type = {}
    
    for entity_type in sorted(entity_types):
        tp = true_positives[entity_type]
        fp = false_positives[entity_type]
        fn = false_negatives[entity_type]
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_by_type[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        print(f"{entity_type:<15} {precision:<12.2%} {recall:<12.2%} {f1:<12.2%} {tp:<8} {fp:<8} {fn:<8}")
    
    # Overall metrics (micro-average)
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("-" * 95)
    print(f"{'OVERALL (Micro)':<15} {overall_precision:<12.2%} {overall_recall:<12.2%} {overall_f1:<12.2%} {total_tp:<8} {total_fp:<8} {total_fn:<8}")
    
    # Macro-average (average of per-class metrics)
    macro_precision = sum(m["precision"] for m in metrics_by_type.values()) / len(metrics_by_type) if metrics_by_type else 0
    macro_recall = sum(m["recall"] for m in metrics_by_type.values()) / len(metrics_by_type) if metrics_by_type else 0
    macro_f1 = sum(m["f1_score"] for m in metrics_by_type.values()) / len(metrics_by_type) if metrics_by_type else 0
    
    print(f"{'OVERALL (Macro)':<15} {macro_precision:<12.2%} {macro_recall:<12.2%} {macro_f1:<12.2%}")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\n✅ Overall Performance (Micro-Average):")
    print(f"   • Precision: {overall_precision:.2%} (of predicted entities, how many were correct)")
    print(f"   • Recall: {overall_recall:.2%} (of true entities, how many were found)")
    print(f"   • F1-Score: {overall_f1:.2%} (harmonic mean)")
    
    print(f"\n✅ Overall Performance (Macro-Average):")
    print(f"   • Precision: {macro_precision:.2%}")
    print(f"   • Recall: {macro_recall:.2%}")
    print(f"   • F1-Score: {macro_f1:.2%}")
    
    print(f"\n📈 Detailed Counts:")
    print(f"   • True Positives: {total_tp}")
    print(f"   • False Positives: {total_fp}")
    print(f"   • False Negatives: {total_fn}")
    print(f"   • Total Predictions: {total_tp + total_fp}")
    print(f"   • Total True Entities: {total_tp + total_fn}")
    
    print(f"\n🎯 Best Performing Entity Types:")
    sorted_by_f1 = sorted(metrics_by_type.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    for entity_type, metrics in sorted_by_f1[:3]:
        print(f"   • {entity_type}: F1={metrics['f1_score']:.2%}, P={metrics['precision']:.2%}, R={metrics['recall']:.2%}")
    
    print(f"\n⚠️  Challenging Entity Types:")
    for entity_type, metrics in sorted_by_f1[-3:]:
        print(f"   • {entity_type}: F1={metrics['f1_score']:.2%}, P={metrics['precision']:.2%}, R={metrics['recall']:.2%}")
    
    # Save results
    results = {
        "overall_metrics": {
            "micro_precision": float(overall_precision),
            "micro_recall": float(overall_recall),
            "micro_f1": float(overall_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1)
        },
        "per_entity_type": {
            entity_type: {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"]
            }
            for entity_type, metrics in metrics_by_type.items()
        },
        "counts": {
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "total_predictions": total_tp + total_fp,
            "total_true_entities": total_tp + total_fn
        },
        "detailed_results": detailed_results
    }
    
    output_path = "/mnt/user-data/outputs/ner_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Results saved to: {output_path}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = evaluate_ner()
    
    print("\n" + "=" * 80)
    print("✅ NER EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n🎯 Key Findings:")
    print(f"   • Overall F1-Score (Micro): {results['overall_metrics']['micro_f1']:.2%}")
    print(f"   • Overall F1-Score (Macro): {results['overall_metrics']['macro_f1']:.2%}")
    print(f"   • Overall Precision: {results['overall_metrics']['micro_precision']:.2%}")
    print(f"   • Overall Recall: {results['overall_metrics']['micro_recall']:.2%}")
    print(f"   • Total Entities Detected: {results['counts']['total_predictions']}")
    print()