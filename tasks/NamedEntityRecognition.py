from transformers import pipeline
nlp = pipeline("ner")
sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very
            close to the Manhattan Bridge which is visible from the window."""
print(nlp(sequence))
#[
#    {'word': 'Hu', 'score': 0.9995632767677307, 'entity': 'I-ORG'},
#    {'word': '##gging', 'score': 0.9915938973426819, 'entity': 'I-ORG'},
#    {'word': 'Face', 'score': 0.9982671737670898, 'entity': 'I-ORG'},
#    {'word': 'Inc', 'score': 0.9994403719902039, 'entity': 'I-ORG'},
#    {'word': 'New', 'score': 0.9994346499443054, 'entity': 'I-LOC'},
#    {'word': 'York', 'score': 0.9993270635604858, 'entity': 'I-LOC'},
#    {'word': 'City', 'score': 0.9993864893913269, 'entity': 'I-LOC'},
#    {'word': 'D', 'score': 0.9825621843338013, 'entity': 'I-LOC'},
#    {'word': '##UM', 'score': 0.936983048915863, 'entity': 'I-LOC'},
#    {'word': '##BO', 'score': 0.8987102508544922, 'entity': 'I-LOC'},
#    {'word': 'Manhattan', 'score': 0.9758241176605225, 'entity': 'I-LOC'},
#    {'word': 'Bridge', 'score': 0.990249514579773, 'entity': 'I-LOC'}
#]



from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge."

# Bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)

print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
#[('[CLS]', 'O'), ('Hu', 'I-ORG'), ('##gging', 'I-ORG'), ('Face', 'I-ORG'), ('Inc', 'I-ORG'), ('.', 'O'), ('is', 'O'), ('a', 'O'), ('company', 'O'), ('based', 'O'), ('in', 'O'), ('New', 'I-LOC'), ('York', 'I-LOC'), ('City', 'I-LOC'), ('.', 'O'), ('Its', 'O'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'), ('D', 'I-LOC'), ('##UM', 'I-LOC'), ('##BO', 'I-LOC'), (',', 'O'), ('therefore', 'O'), ('very', 'O'), ('##c', 'O'), ('##lose', 'O'), ('to', 'O'), ('the', 'O'), ('Manhattan', 'I-LOC'), ('Bridge', 'I-LOC'), ('.', 'O'), ('[SEP]', 'O')]