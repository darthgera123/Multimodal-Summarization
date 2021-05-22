# Multimodal Article Summarization
This project aims to perform multimodal article summarization using pretrained models. This project was done with [Prof Vasudev Varma](https://scholar.google.co.in/citations?user=9OFvbfcAAAAJ&hl=en) and [Balaji Vasan Srinivasan (Adobe)](https://research.adobe.com/person/balaji-vasan-srinivasan/). We leverage Pretrained models to perform summarization of articles. Unlike previous methods, our method takes both the article and image as input and the output is a text summary

[Detailed Report](https://docs.google.com/document/d/1BCK9JDG0YjxhSX4YozMhr6rnK8gk2LmfkHA_zcpnw1c/edit?usp=sharing) 

## Method
In this codebase we leverage OSCAR as our pretrained encoder and GPT2 as our pretrained decoder. We use nucleus sampling to generate text. OSCAR constructs a shared image-text embedding and minimizes distance b/w the Faster-RCNN features of the object and the corresponding word embedding. However you can replace OSCAR with any other visio-linguistic transformer like LXMERT, UNITER,etc. Similarly you can replace GPT2LMHead with any other LM head to generate logits. The components are extremely modular.

## Installation
This codebase uses vilio library as our backbone which inturn uses `huggingface-3.5.0`. To install simply do
`pip3 install -r requirements.txt`. Further instructions are present in `GETTING_STARTED.md`
To run this code, simply run `bash exp.sh`. 
