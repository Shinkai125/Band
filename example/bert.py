"""
@file: bert.py
@time: 2020-09-23 20:36:05
"""
from model.bert import BERT

if __name__ == '__main__':
    model = BERT(maxlen=200, vocab_size=21128, embed_dim=768, num_layers=12, num_heads=12, feed_forward_dim=3072)
    model.summary(line_length=150)
    for i in range(len(model.layers)):
        print(model.get_layer(index=i).output)
