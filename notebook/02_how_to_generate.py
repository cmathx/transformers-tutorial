import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')

############1. greedy search############
# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


#Output:
#----------------------------------------------------------------------------------------------------
#I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll ever be able to walk with my dog.
#
#I'm not sure if I'll


############2. beam search############
# activate beam search and early_stopping
beam_output = model.generate(
    input_ids,  
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
#Output:
#----------------------------------------------------------------------------------------------------
#I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
#
#I've been thinking about this for a while now, and I think it's time for me to take a break

# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
#Output:
#----------------------------------------------------------------------------------------------------
#0: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
#
#I've been thinking about this for a while now, and I think it's time for me to take a break
#1: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
#
#I've been thinking about this for a while now, and I think it's time for me to get back to
#2: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with her again.
#
#I've been thinking about this for a while now, and I think it's time for me to take a break
#3: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with her again.
#
#I've been thinking about this for a while now, and I think it's time for me to get back to
#4: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
#
#I've been thinking about this for a while now, and I think it's time for me to take a step


############3.1 sampling############
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0,
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
#Output:
#----------------------------------------------------------------------------------------------------
#I enjoy walking with my cute dog, but I don't like to be at home too much. I also find it a bit weird when I'm out shopping. I am always away from my house a lot, but I do have a few friends

############3.2 Top-K sampling############
tf.random.set_seed(0)

# set top_k to 50
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
#Output:
#----------------------------------------------------------------------------------------------------
#I enjoy walking with my cute dog. It's so good to have an environment where your dog is available to share with you and we'll be taking care of you.
#
#We hope you'll find this story interesting!
#
#I am from

############3.3 Top-p (nucleus) sampling############
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
#Output:
#----------------------------------------------------------------------------------------------------
#I enjoy walking with my cute dog. He will never be the same. I watch him play.
#
#
#Guys, my dog needs a name. Especially if he is found with wings.
#
#
#What was that? I had a lot of

# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
#Output:
#----------------------------------------------------------------------------------------------------
#0: I enjoy walking with my cute dog. It's so good to have the chance to walk with a dog. But I have this problem with the dog and how he's always looking at us and always trying to make me see that I can do something
#1: I enjoy walking with my cute dog, she loves taking trips to different places on the planet, even in the desert! The world isn't big enough for us to travel by the bus with our beloved pup, but that's where I find my love
#2: I enjoy walking with my cute dog and playing with our kids," said David J. Smith, director of the Humane Society of the US.
#
#"So as a result, I've got more work in my time," he said.

