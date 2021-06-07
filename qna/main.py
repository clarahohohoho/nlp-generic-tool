from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def main(context, question):
    
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # a) Get predictions
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)

    # return res['answer'], res['score']
    return res['answer']