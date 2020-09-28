import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

torch.set_grad_enabled(False)

# Store the model we want to use
MODEL_NAME = "bert-base-cased"

# We need to create the model and tokenizer
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

############1. basic usage############
# Tokens comes from a process that splits the input into sub-entities with interesting linguistic properties. 
tokens = tokenizer.tokenize("This is an input example")
print("Tokens: {}".format(tokens))

# This is not sufficient for the model, as it requires integers as input, 
# not a problem, let's convert tokens to ids.
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))

# Add the required special tokens
tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)

# We need to convert to a Deep Learning framework specific format, let's use PyTorch for now.
tokens_pt = torch.tensor([tokens_ids])
print("Tokens PyTorch: {}".format(tokens_pt))

# Now we're ready to go through BERT with out input
outputs, pooled = model(tokens_pt)
print("Token wise output: {}, Pooled output: {}".format(outputs.shape, pooled.shape))


#Tokens: ['This', 'is', 'an', 'input', 'example']
#Tokens id: [1188, 1110, 1126, 7758, 1859]
#Tokens PyTorch: tensor([[ 101, 1188, 1110, 1126, 7758, 1859,  102]])
#Token wise output: torch.Size([1, 7, 768]), Pooled output: torch.Size([1, 768])

############1. simple usage############
# tokens = tokenizer.tokenize("This is an input example")
# tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
# tokens_pt = torch.tensor([tokens_ids])

# This code can be factored into one-line as follow
tokens_pt2 = tokenizer("This is an input example", return_tensors="pt")

for key, value in tokens_pt2.items():
    print("{}:\n\t{}".format(key, value))

outputs2, pooled2 = model(**tokens_pt2)
print("Difference with previous code: ({}, {})".format((outputs2 - outputs).sum(), (pooled2 - pooled).sum()))
#input_ids:
#	tensor([[ 101, 1188, 1110, 1126, 7758, 1859,  102]])
#token_type_ids:
#	tensor([[0, 0, 0, 0, 0, 0, 0]])
#attention_mask:
#	tensor([[1, 1, 1, 1, 1, 1, 1]])
#Difference with previous code: (0.0, 0.0)