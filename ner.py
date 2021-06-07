import spacy

def ner(text):

    dictt = {'person': [], 'org':[], 'date':[], 'loc':[], 'pdt':[], 'event':[]}


    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for ent in doc.ents:

        if ent.label_ == 'PERSON' and ent.text not in dictt['person']:
            dictt['person'].append(ent.text)
        elif ent.label_ == 'ORG' and ent.text not in dictt['org']:
            dictt['org'].append(ent.text)
        elif ent.label_ == 'DATE' and ent.text not in dictt['date']:
            dictt['date'].append(ent.text)
        elif ent.label_ == 'LOC' and ent.text not in dictt['loc']:
            dictt['loc'].append(ent.text)
        elif ent.label_ == 'PRODUCT' and ent.text not in dictt['pdt']:
            dictt['pdt'].append(ent.text)
        elif ent.label_ == 'EVENT' and ent.text not in dictt['event']:
            dictt['event'].append(ent.text)

    return dictt

