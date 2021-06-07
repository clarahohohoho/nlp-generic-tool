from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import numpy as np

def weapon_main(context, question):

    res = []

    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)

    start_values, start_indices = torch.topk(torch.nn.functional.softmax(answer_start_scores, dim = 1), 3)  # Get the most likely beginning of answer with the argmax of the score
    end_values, end_indices = torch.topk(torch.nn.functional.softmax(answer_end_scores, dim = 1), 3)  # Get the most likely end of answer with the argmax of the score
    end_indices += 1

    for i in range(3):
        start = start_indices[0][i]
        end = end_indices[0][i]
        # if start < end:
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end]))
        # if end < start:
        #     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[end:start]))
        print(end_values[0][i].detach().numpy(), answer)
        if end_values[0][i].detach().numpy() > 0.2 and answer!="" and '[CLS]' not in answer and '[SEP]' not in answer:
            res.append(answer)
    return res