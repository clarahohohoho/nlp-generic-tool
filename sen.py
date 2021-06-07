import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

def sen(text):

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(text)
    print(doc._.polarity)

    return doc._.polarity