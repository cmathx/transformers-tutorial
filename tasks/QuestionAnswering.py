from transformers import pipeline
nlp = pipeline("question-answering")
context = r"""
    Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
    question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
    a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
    """
result = nlp(question="What is extractive question answering?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
#Answer: 'the task of extracting an answer from a text given a question.', score: 0.6226, start: 34, end: 96

result = nlp(question="What is a good example of a question answering dataset?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
#Answer: 'SQuAD dataset,', score: 0.5053, start: 147, end: 161



from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
       🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
       architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
       Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
       TensorFlow 2.0 and PyTorch.
       """

questions = [
     "How many pretrained models are available in 🤗 Transformers?",
     "What does 🤗 Transformers provide?",
     "🤗 Transformers provides interoperability between which frameworks?",
 ]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}")
#Question: How many pretrained models are available in 🤗 Transformers?
#Answer: over 32 +
#Question: What does 🤗 Transformers provide?
#Answer: general - purpose architectures
#Question: 🤗 Transformers provides interoperability between which frameworks?
#Answer: tensorflow 2 . 0 and pytorch