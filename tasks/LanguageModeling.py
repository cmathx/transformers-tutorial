###1) Masked Language Modeling
from transformers import pipeline
nlp = pipeline("fill-mask")

from pprint import pprint
pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
#[{'score': 0.1792745739221573,
#  'sequence': '<s>HuggingFace is creating a tool that the community uses to '
#              'solve NLP tasks.</s>',
#  'token': 3944,
#  'token_str': 'Ġtool'},
# {'score': 0.11349421739578247,
#  'sequence': '<s>HuggingFace is creating a framework that the community uses '
#              'to solve NLP tasks.</s>',
#  'token': 7208,
#  'token_str': 'Ġframework'},
# {'score': 0.05243554711341858,
#  'sequence': '<s>HuggingFace is creating a library that the community uses to '
#              'solve NLP tasks.</s>',
#  'token': 5560,
#  'token_str': 'Ġlibrary'},
# {'score': 0.03493533283472061,
#  'sequence': '<s>HuggingFace is creating a database that the community uses '
#              'to solve NLP tasks.</s>',
#  'token': 8503,
#  'token_str': 'Ġdatabase'},
# {'score': 0.02860250137746334,
#  'sequence': '<s>HuggingFace is creating a prototype that the community uses '
#              'to solve NLP tasks.</s>',
#  'token': 17715,
#  'token_str': 'Ġprototype'}]



from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
#Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
#Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
#Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
#Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
#Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.



###2) Causal Language Modeling
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

sequence = f"Hugging Face is based in DUMBO, New York City, and "

input_ids = tokenizer.encode(sequence, return_tensors="pt")

# get logits of last hidden state
next_token_logits = model(input_ids).logits[:, -1, :]

# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
#Hugging Face is based in DUMBO, New York City, and has