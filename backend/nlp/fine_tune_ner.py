import spacy
from spacy.training.example import Example
from pathlib import Path
import random

# =========================
# 1. EXPANDED TRAINING DATA
# =========================
TRAIN_DATA = [
    # Football/Soccer Players and Teams
    ("Lionel Messi scored a goal for Argentina", 
     {"entities": [(0, 12, "PLAYER"), (31, 40, "TEAM")]}),
    
    ("Cristiano Ronaldo joined Al Nassr", 
     {"entities": [(0, 17, "PLAYER"), (26, 34, "TEAM")]}),
    
    ("Neymar plays for Brazil national team", 
     {"entities": [(0, 6, "PLAYER"), (17, 38, "TEAM")]}),
    
    ("Kylian Mbappé is a striker for Paris Saint-Germain", 
     {"entities": [(0, 14, "PLAYER"), (32, 51, "TEAM")]}),
    
    ("Erling Haaland transferred to Manchester City", 
     {"entities": [(0, 14, "PLAYER"), (30, 46, "TEAM")]}),
    
    ("Mohamed Salah represents Egypt in international matches", 
     {"entities": [(0, 13, "PLAYER"), (25, 30, "TEAM")]}),
    
    ("Kevin De Bruyne is the captain of Belgium", 
     {"entities": [(0, 15, "PLAYER"), (34, 41, "TEAM")]}),
    
    ("Harry Kane scored for England at the World Cup", 
     {"entities": [(0, 10, "PLAYER"), (22, 29, "TEAM")]}),
    
    ("Robert Lewandowski plays for Poland", 
     {"entities": [(0, 18, "PLAYER"), (29, 35, "TEAM")]}),
    
    ("Luka Modrić captains Real Madrid", 
     {"entities": [(0, 12, "PLAYER"), (21, 32, "TEAM")]}),
    
    # Add more diverse examples
    ("Barcelona signed a new contract with Pedri", 
     {"entities": [(0, 9, "TEAM"), (37, 42, "PLAYER")]}),
    
    ("Liverpool defeated Manchester United in the derby", 
     {"entities": [(0, 9, "TEAM"), (19, 37, "TEAM")]}),
    
    ("The match between France and Germany was exciting", 
     {"entities": [(18, 24, "TEAM"), (29, 36, "TEAM")]}),
    
    ("Karim Benzema won the Ballon d'Or while playing for France", 
     {"entities": [(0, 13, "PLAYER"), (52, 58, "TEAM")]}),
    
    ("Sergio Ramos defended for Spain in the final", 
     {"entities": [(0, 12, "PLAYER"), (26, 31, "TEAM")]}),
    
    # Add negative examples (common words that should NOT be entities)
    ("The ball went into the goal after a great pass", 
     {"entities": []}),
    
    ("The stadium was full of fans cheering loudly", 
     {"entities": []}),
    
    ("The referee made a controversial decision during the match", 
     {"entities": []}),
]

# =========================
# 2. LOAD MODEL PRETRAINED
# =========================
print("📦 Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

ner = nlp.get_pipe("ner")

# =========================
# 3. ADD CUSTOM LABELS
# =========================
print("🏷️  Adding custom labels...")
ner.add_label("PLAYER")
ner.add_label("TEAM")

# =========================
# 4. TRAINING (FINE-TUNING)
# =========================
print("🎯 Starting fine-tuning...")

# Disable other pipelines during training
other_pipes = [p for p in nlp.pipe_names if p != "ner"]

with nlp.disable_pipes(*other_pipes):
    # Initialize the optimizer
    optimizer = nlp.resume_training()
    
    # Training parameters
    n_iter = 30  # Increased iterations for better learning
    
    for epoch in range(n_iter):
        # Shuffle training data
        random.shuffle(TRAIN_DATA)
        losses = {}
        
        # Batch training examples
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
        
        # Print progress every 5 iterations
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch + 1}/{n_iter} - Loss: {losses.get('ner', 0):.2f}")

# =========================
# 5. SAVE MODEL
# =========================
output_dir = Path("nlp/fine_tuned_ner")
output_dir.mkdir(parents=True, exist_ok=True)

nlp.to_disk(output_dir)

print("\n✅ Fine-tuning complete!")
print(f"📁 Model saved to: {output_dir.absolute()}")

# =========================
# 6. TEST THE MODEL
# =========================
print("\n🧪 Testing the fine-tuned model...\n")

test_sentences = [
    "Lionel Messi scored a goal for Argentina",
    "Cristiano Ronaldo joined Al Nassr",
    "Manchester United played against Liverpool",
    "Neymar represents Brazil in the World Cup",
]

for sentence in test_sentences:
    doc = nlp(sentence)
    print(f"Text: {sentence}")
    if doc.ents:
        for ent in doc.ents:
            print(f"  - {ent.text:20} → {ent.label_}")
    else:
        print("  (no entities found)")
    print()