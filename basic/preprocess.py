"""https://huggingface.co/transformers/preprocessing.html"""


"""
"" tokenizer(batch_sentences, padding, truncation, max_length)
""     padding:
""	       True/longest    - to pad to the longest sequence in the batch
""	       False           - to pad to a length specified by the max_length
""	       max_length      - no padding
""     truncation:
""	       True/only_first - truncate to a maximum length specified by the max_length argument or the maximum length accepted by the model if no max_length is
""							 provided (max_length=None). This will only truncate the first sentence of a pair if a pair of sequence (or a batch of pairs of sequences) is provided.
""	       only_second     - truncate to a maximum length specified by the max_length argument or the maximum length accepted by the model if no max_length is
""							 provided (max_length=None). This will only truncate the second sentence of a pair if a pair of sequence (or a batch of pairs of sequences) is provided.
""	       longest_first   - truncate to a maximum length specified by the max_length argument or the maximum length accepted by the model if no max_length is
""	       	                 provided (max_length=None). This will truncate token by token, removing a token from the longest sequence in the pair until the proper length is reached.
""	       False           - no truncation
""	   max_length:
""		   control the length of the padding/truncation
""	   
"""

from transformers import AutoTokenizer

#########1. encode only one sentence#########
batch_sentences = ["Hello I'm a single sentence",
				   "And another sentence",
				   "And the very very last one"]
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
batch_encoded_input = tokenizer(batch_sentences)
#batch_encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(batch_encoded_input)
"""
{'input_ids': [[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
               [101, 1262, 1330, 5650, 102],
               [101, 1262, 1103, 1304, 1304, 1314, 1141, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]}
 """

 tokenizer.decode(encoded_input["input_ids"])


#########2. encoder two sentences#########
batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
                             "And I should be encoded with the second sentence",
                             "And I go with the very last one"]
encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
#batch_encoded_inpuuts = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_inputs)
"""
{'input_ids': [[101, 8667, 146, 112, 182, 170, 1423, 5650, 102, 146, 112, 182, 170, 5650, 1115, 2947, 1114, 1103, 1148, 5650, 102],
               [101, 1262, 1330, 5650, 102, 1262, 146, 1431, 1129, 12544, 1114, 1103, 1248, 5650, 102],
               [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 1262, 146, 1301, 1114, 1103, 1304, 1314, 1141, 102]],
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
"""

