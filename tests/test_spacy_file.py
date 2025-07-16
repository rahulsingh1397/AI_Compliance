import spacy
import sys
import os

# Define the output path for the test results
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_file = os.path.join(project_root, 'tests', 'test_results', 'spacy_file_test_output.txt')

with open(output_file, 'w') as f:
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
