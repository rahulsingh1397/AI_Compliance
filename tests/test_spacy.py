import sys
import spacy
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"SpaCy version: {spacy.__version__}")
print(f"SpaCy path: {Path(spacy.__file__).parent}")

try:
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully!")
    
    # Test the model on a simple sentence
    text = "Apple is looking at buying U.K. startup for $1 billion"
    doc = nlp(text)
    
    print("\nTokenization and POS tagging test:")
    for token in doc:
        print(f"{token.text:{15}} {token.pos_:{10}} {token.dep_:{10}}")
    
    print("\nNamed Entity Recognition test:")
    for ent in doc.ents:
        print(f"{ent.text:{15}} {ent.label_}")
    
    print("\nSpaCy test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
