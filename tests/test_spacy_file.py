import spacy
import sys

with open('spacy_test_output.txt', 'w') as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"SpaCy version: {spacy.__version__}\n")
    
    try:
        f.write("Loading spaCy model...\n")
        nlp = spacy.load("en_core_web_sm")
        f.write("SpaCy model loaded successfully!\n")
        
        # Test the model on a simple sentence
        text = "Apple is looking at buying U.K. startup for $1 billion"
        doc = nlp(text)
        
        f.write("\nTokenization test:\n")
        for token in doc:
            f.write(f"{token.text} ")
        
        f.write("\n\nNamed Entity Recognition test:\n")
        for ent in doc.ents:
            f.write(f"{ent.text}: {ent.label_}\n")
        
        f.write("\nSpaCy test completed successfully!\n")
        
    except Exception as e:
        f.write(f"Error: {e}\n")
