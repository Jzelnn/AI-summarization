"""
Topic Detection Accuracy Evaluation
Tests the keyword-based topic classification system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp.pipeline import detect_topic
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json

# Test dataset with ground truth labels
TEST_DATASET = [
    # SPORTS texts
    {
        "text": "Lionel Messi scored a stunning goal as Argentina defeated Brazil 2-1 in the Copa America final. The match was held in Buenos Aires stadium.",
        "label": "sports"
    },
    {
        "text": "Manchester United signed a new striker in the transfer window. The player will join the team for the Premier League season.",
        "label": "sports"
    },
    {
        "text": "The basketball championship game went into overtime. The team's star player scored 35 points to secure the victory.",
        "label": "sports"
    },
    {
        "text": "The tennis tournament featured incredible matches. Djokovic defeated his opponent in straight sets to win the trophy.",
        "label": "sports"
    },
    {
        "text": "The football coach announced the starting lineup for tomorrow's crucial match. The team needs a win to advance in the tournament.",
        "label": "sports"
    },
    {
        "text": "The Olympic athlete broke the world record in the 100-meter sprint. The stadium erupted in celebration.",
        "label": "sports"
    },
    {
        "text": "The soccer league announced new rules for the upcoming season. Teams will have five substitutions allowed per game.",
        "label": "sports"
    },
    {
        "text": "Cricket fans celebrated as their team won the World Cup final. The captain praised the team's dedication and effort.",
        "label": "sports"
    },
    {
        "text": "The NFL draft saw several surprising picks. Teams are building their rosters for the next championship run.",
        "label": "sports"
    },
    {
        "text": "The Formula 1 race was intense with multiple overtakes. The driver secured pole position for the Grand Prix.",
        "label": "sports"
    },
    {
        "text": "The hockey team won in a shootout after a thrilling game. The goalie made several spectacular saves.",
        "label": "sports"
    },
    {
        "text": "The boxing match lasted all 12 rounds. The judges scored it unanimously for the champion.",
        "label": "sports"
    },
    {
        "text": "The golf tournament at Augusta was won by a record margin. The player had an incredible final round.",
        "label": "sports"
    },
    {
        "text": "The volleyball team qualified for the finals. They defeated the top-ranked opponent in five sets.",
        "label": "sports"
    },
    {
        "text": "The baseball pitcher threw a no-hitter. It was one of the greatest performances in the league's history.",
        "label": "sports"
    },
    
    # POLITICS texts
    {
        "text": "The president announced new policies during the parliamentary session. The legislation will be voted on next week by congress.",
        "label": "politics"
    },
    {
        "text": "The election results showed a clear victory for the democratic party. Voter turnout was higher than expected.",
        "label": "politics"
    },
    {
        "text": "The senate passed the new bill with bipartisan support. The prime minister praised the collaboration.",
        "label": "politics"
    },
    {
        "text": "Political tensions rose as the government faced criticism over economic policies. The opposition called for reforms.",
        "label": "politics"
    },
    {
        "text": "The minister of foreign affairs met with international leaders. They discussed trade agreements and diplomatic relations.",
        "label": "politics"
    },
    {
        "text": "Campaign rallies drew large crowds as candidates presented their platforms. The presidential race remains competitive.",
        "label": "politics"
    },
    {
        "text": "The Supreme Court ruled on the constitutional challenge. Legal experts analyzed the implications of the decision.",
        "label": "politics"
    },
    {
        "text": "Parliament debated the new legislation for hours. Senators from both parties voiced strong opinions.",
        "label": "politics"
    },
    {
        "text": "The governor announced a state of emergency. Political leaders coordinated the response to the crisis.",
        "label": "politics"
    },
    {
        "text": "Diplomatic negotiations continued as nations worked toward a peace agreement. The ambassador led the discussions.",
        "label": "politics"
    },
    {
        "text": "The political scandal dominated headlines. Lawmakers called for an independent investigation.",
        "label": "politics"
    },
    {
        "text": "The mayor proposed a new budget for the city. Council members reviewed the fiscal policy proposals.",
        "label": "politics"
    },
    {
        "text": "International sanctions were imposed following the vote. The United Nations secretary addressed member states.",
        "label": "politics"
    },
    {
        "text": "The referendum results showed public support for the constitutional amendment. Democracy prevailed.",
        "label": "politics"
    },
    {
        "text": "Congressional hearings examined the executive branch's actions. Oversight committees questioned government officials.",
        "label": "politics"
    },
    
    # TECHNOLOGY texts
    {
        "text": "The new AI model demonstrates impressive capabilities in natural language processing. Machine learning algorithms continue to advance.",
        "label": "tech"
    },
    {
        "text": "The tech startup raised millions in funding. Their innovative software platform attracted major investors.",
        "label": "tech"
    },
    {
        "text": "Artificial intelligence is transforming how companies analyze data. Cloud computing infrastructure supports these innovations.",
        "label": "tech"
    },
    {
        "text": "The software update includes new features and security patches. Developers worked on improving the user interface.",
        "label": "tech"
    },
    {
        "text": "Quantum computing research achieved a major breakthrough. Scientists published their findings in a technical journal.",
        "label": "tech"
    },
    {
        "text": "The mobile application reached one million downloads. Users praised the intuitive design and functionality.",
        "label": "tech"
    },
    {
        "text": "Cybersecurity experts warned about new vulnerabilities. Companies need to update their digital infrastructure.",
        "label": "tech"
    },
    {
        "text": "The programming language gained popularity among developers. Open-source contributions improved the codebase.",
        "label": "tech"
    },
    {
        "text": "Virtual reality technology advanced with new hardware releases. Gaming applications showcase immersive experiences.",
        "label": "tech"
    },
    {
        "text": "The database system handles petabytes of information. Big data analytics drive business intelligence decisions.",
        "label": "tech"
    },
    {
        "text": "Blockchain technology promises to revolutionize financial systems. Cryptocurrency adoption continues to grow.",
        "label": "tech"
    },
    {
        "text": "The robotics company unveiled autonomous navigation systems. Sensors and algorithms enable precise movement.",
        "label": "tech"
    },
    {
        "text": "5G networks are being deployed in major cities. Telecommunications infrastructure supports faster connectivity.",
        "label": "tech"
    },
    {
        "text": "The algorithm optimizes performance through neural networks. Deep learning models achieve state-of-the-art results.",
        "label": "tech"
    },
    {
        "text": "Internet of Things devices are becoming ubiquitous. Smart home technology integrates with cloud platforms.",
        "label": "tech"
    },
    
    # GENERAL texts
    {
        "text": "The weather forecast predicts rain for the weekend. Temperatures will remain mild throughout the week.",
        "label": "general"
    },
    {
        "text": "The local community organized a charity event. Volunteers helped raise funds for the children's hospital.",
        "label": "general"
    },
    {
        "text": "A new restaurant opened in the downtown area. The chef specializes in Mediterranean cuisine.",
        "label": "general"
    },
    {
        "text": "The museum exhibition features contemporary art. Visitors can view paintings and sculptures from local artists.",
        "label": "general"
    },
    {
        "text": "Public transportation schedules changed this month. Commuters should check the updated bus and train times.",
        "label": "general"
    },
    {
        "text": "The annual festival attracts thousands of visitors. Music performances and food vendors line the streets.",
        "label": "general"
    },
    {
        "text": "Real estate prices increased in the metropolitan area. Home buyers face competitive market conditions.",
        "label": "general"
    },
    {
        "text": "The education system implemented new curriculum standards. Teachers attended professional development workshops.",
        "label": "general"
    },
    {
        "text": "Healthcare workers received recognition for their dedication. The hospital announced expanded services.",
        "label": "general"
    },
    {
        "text": "Environmental conservation efforts focus on reducing waste. Recycling programs encourage sustainable practices.",
        "label": "general"
    },
    {
        "text": "The book became a bestseller within weeks of release. Readers praised the author's storytelling ability.",
        "label": "general"
    },
    {
        "text": "Traffic congestion increased during rush hour. City planners proposed infrastructure improvements.",
        "label": "general"
    },
    {
        "text": "The theater production received rave reviews. Actors delivered powerful performances on opening night.",
        "label": "general"
    },
    {
        "text": "Consumer spending patterns shifted during the holiday season. Retailers reported strong sales figures.",
        "label": "general"
    },
    {
        "text": "The university announced scholarship opportunities. Students can apply for financial aid programs.",
        "label": "general"
    },
]


def evaluate_topic_detection():
    """
    Evaluate topic detection accuracy with detailed metrics
    """
    print("=" * 80)
    print("TOPIC DETECTION ACCURACY EVALUATION")
    print("=" * 80)
    
    # Get predictions
    true_labels = []
    predicted_labels = []
    
    print(f"\n📊 Testing on {len(TEST_DATASET)} documents...\n")
    
    for i, sample in enumerate(TEST_DATASET, 1):
        text = sample["text"]
        true_label = sample["label"]
        predicted_label = detect_topic(text)
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        
        # Show some examples
        if i <= 5 or predicted_label != true_label:
            status = "✅" if predicted_label == true_label else "❌"
            print(f"{status} Sample {i}:")
            print(f"   Text: {text[:80]}...")
            print(f"   True: {true_label} | Predicted: {predicted_label}")
            if predicted_label != true_label:
                print(f"   ⚠️  MISCLASSIFICATION")
            print()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Per-class metrics
    labels = ['sports', 'politics', 'tech', 'general']
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, average=None, zero_division=0
    )
    
    # Macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, average='macro', zero_division=0
    )
    
    # Weighted average
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    # Print results
    print("=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    print(f"\n✅ Overall Accuracy: {accuracy:.2%}\n")
    
    print("=" * 80)
    print("📈 PER-CATEGORY PERFORMANCE")
    print("=" * 80)
    print(f"\n{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i, label in enumerate(labels):
        print(f"{label.upper():<15} {precision[i]:<12.2%} {recall[i]:<12.2%} {f1[i]:<12.2%} {int(support[i]):<10}")
    
    print("-" * 80)
    print(f"{'MACRO AVG':<15} {precision_macro:<12.2%} {recall_macro:<12.2%} {f1_macro:<12.2%}")
    print(f"{'WEIGHTED AVG':<15} {precision_weighted:<12.2%} {recall_weighted:<12.2%} {f1_weighted:<12.2%}")
    print()
    
    # Confusion matrix
    print("=" * 80)
    print("🔢 CONFUSION MATRIX")
    print("=" * 80)
    print("\nRows = True Label | Columns = Predicted Label\n")
    print(f"{'':>12}", end='')
    for label in labels:
        print(f"{label.upper():<12}", end='')
    print()
    print("-" * 80)
    
    for i, true_label in enumerate(labels):
        print(f"{true_label.upper():<12}", end='')
        for j, pred_label in enumerate(labels):
            print(f"{cm[i][j]:<12}", end='')
        print()
    print()
    
    # Category-specific insights
    print("=" * 80)
    print("💡 CATEGORY INSIGHTS")
    print("=" * 80)
    
    for i, label in enumerate(labels):
        print(f"\n{label.upper()}:")
        print(f"  • Precision: {precision[i]:.2%} (of predictions, how many were correct)")
        print(f"  • Recall: {recall[i]:.2%} (of true {label} texts, how many were found)")
        print(f"  • F1-Score: {f1[i]:.2%} (harmonic mean of precision and recall)")
        print(f"  • Test samples: {int(support[i])}")
        
        # Find main confusion sources
        misclassified_as = {}
        for j, other_label in enumerate(labels):
            if i != j and cm[i][j] > 0:
                misclassified_as[other_label] = cm[i][j]
        
        if misclassified_as:
            print(f"  • Misclassified as:", end='')
            for other, count in sorted(misclassified_as.items(), key=lambda x: x[1], reverse=True):
                print(f" {other}({count})", end='')
            print()
    
    # Save results to JSON
    results = {
        "overall_accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_category": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(len(labels))
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    
    output_path = "/mnt/user-data/outputs/topic_detection_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Results saved to: {output_path}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = evaluate_topic_detection()
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n🎯 Key Findings:")
    print(f"   • Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"   • Macro F1-Score: {results['macro_f1']:.2%}")
    print(f"   • Best Category: {max(results['per_category'].items(), key=lambda x: x[1]['f1_score'])[0].upper()}")
    print(f"   • Weakest Category: {min(results['per_category'].items(), key=lambda x: x[1]['f1_score'])[0].upper()}")
    print()