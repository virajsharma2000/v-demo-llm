import spacy

nlp = spacy.blank('en')

doc = nlp("Dr.Einstine discovered theory of relativity")

tokens = []

for token in doc:
    tokens.append(str(token))

print(tokens)