import language_tool_python
import spacy

# Create a LanguageTool object for French
tool = language_tool_python.LanguageTool('fr')
nlp = spacy.load("fr_core_news_sm")
doc = nlp("J'ai apris le fransais.")

def correct_french_spelling(text):
    # Check for spelling and grammar errors
    matches = tool.check(text)
    print (matches)
    # Correct spelling and grammar errors
    corrected_text = tool.correct(text)

    return corrected_text

# Example usage
input_text = "J'ai apris le fransais."
corrected_text = correct_french_spelling(input_text)
print(f"language_tool_python: {corrected_text}")

for token in doc:
    print(token.text, token.lemma_, token.pos_)


print(corrected_text)
