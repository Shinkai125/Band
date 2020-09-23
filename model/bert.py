"""
@file: bert.py
@time: 2020-09-23 15:27:12
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers.TokenAndPositionEmbedding import BERTEmbedding
from layers.TransformerBlock import TransformerBlock


def BERT(vocab_size=2000, maxlen=100, embed_dim=256, num_heads=2, feed_forward_dim=256, num_layers=1):
    """
    :param num_layers: Transformer Block number
    :param vocab_size: Vocabulary size
    :param maxlen: Max sequence size
    :param embed_dim: Embedding size for each token
    :param num_heads: Number of attention heads
    :param feed_forward_dim: Hidden layer size in feed forward network inside transformer
    :return: tf.keras.Model
    """
    token_input = layers.Input(shape=(maxlen,), dtype=tf.int32, name='Input-Token')
    segment_input = layers.Input(shape=(maxlen,), dtype=tf.int32, name='Input-Segment')
    mask_input = layers.Input(shape=(maxlen,), dtype=tf.int32, name='Input-Masked')

    embedding = BERTEmbedding(maxlen, vocab_size, embed_dim)([token_input, segment_input])
    for i in range(num_layers):
        if i == 0:
            sequence_output = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(embedding)
        else:
            sequence_output = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(sequence_output)
    pooled_output = layers.GlobalAveragePooling1D()(sequence_output)
    model = keras.Model(inputs=[token_input, segment_input], outputs=[sequence_output, pooled_output])
    return model


if __name__ == '__main__':
    model = BERT(maxlen=200, vocab_size=21128, embed_dim=768, num_layers=12, num_heads=12, feed_forward_dim=3072)
    model.summary(line_length=150)
