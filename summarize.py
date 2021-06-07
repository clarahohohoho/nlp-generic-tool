from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
  
def summarize(text):
    
    model_name = "sshleifer/distilbart-cnn-12-6"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
    res = summarizer(text, max_length=130, min_length=30)

    return res[0]['summary_text']