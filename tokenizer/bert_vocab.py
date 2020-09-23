"""
@file: bert_vocab.py
@time: 2020-09-23 16:06:29
"""
import codecs


def load_vocabulary(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
