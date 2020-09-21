import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from layers.TransformerBlock import TransformerBlock


def Transformer(vocab_size=2000, maxlen=100, embed_dim=256, num_heads=2, feed_forward_dim=256, output_dim=2):
    """
    :param vocab_size: Vocabulary size
    :param maxlen: Max sequence size
    :param embed_dim: Embedding size for each token
    :param num_heads: Number of attention heads
    :param feed_forward_dim: Hidden layer size in feed forward network inside transformer
    :param output_dim: output dim
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(embedding)
    outputs = layers.Dense(output_dim)(transformer_block)
    model = keras.Model(inputs=inputs, outputs=[outputs])
    return model
