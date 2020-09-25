import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from layers.TransformerBlock import TransformerBlock


def Transformer(
        vocab_size=2000,
        maxlen=100,
        embed_dim=256,
        num_heads=2,
        feed_forward_dim=256,
):
    """
    :param vocab_size: Vocabulary size
    :param maxlen: Max sequence size
    :param embed_dim: Embedding size for each token
    :param num_heads: Number of attention heads
    :param feed_forward_dim: Hidden layer size in feed forward network inside transformer
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(
        inputs
    )
    sequence_output = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(
        embedding
    )
    pooled_output = layers.GlobalAveragePooling1D()(sequence_output)
    model = keras.Model(
        inputs=inputs, outputs=[sequence_output, pooled_output]
    )
    return model


if __name__ == "__main__":
    model = Transformer()
    model.summary()
