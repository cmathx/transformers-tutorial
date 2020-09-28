0）Byte-Pair Encoding
基于BPE的编码方式，每次总是合并选取频次最多的subword，直到满足相应规则合并结束；
词表大小：基础的词语+合并的次数，在gpt模型中使用的词表大小为40478（478个基础的词语、40000次合并）
模型：GPT、Roberta、XLM、FlauBERT

1）Byte-level BPE
在BPE编码的基础上，以字节为单位构建基础的词语。在gpt-2中使用的词表大小为50257（256个基础的词语、50000次合并、一个<eos>）
模型： GPT-2

2）WordPiece
合并规则：基于概率生成新的subword；
模型： BERT、DistilBERT、Electra

3）Unigram

4）SentencePiece
模型：XLNET