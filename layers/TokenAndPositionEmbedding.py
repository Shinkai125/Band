import tensorflow as tf
from tensorflow.keras import layers


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="word_embeddings")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name="position_embeddings")

    def call(self, x, **kwargs):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class BERTEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, dropout_rate=0.1):
        super(BERTEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="word_embeddings")
        self.segment_emb = layers.Embedding(input_dim=2, output_dim=embed_dim, name="segment_embeddings")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name="position_embeddings")
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, **kwargs):
        token_input, segment_input = x
        x = self.token_emb(token_input) + self.segment_emb(segment_input)

        maxlen = tf.shape(token_input)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)

        return self.layernorm(self.dropout(x + positions))


if __name__ == '__main__':
    input1 = layers.Input(shape=(200,))
    input2 = layers.Input(shape=(200,))
    embedding = BERTEmbedding(maxlen=200, vocab_size=21128, embed_dim=768)([input1, input2])
    model = tf.keras.Model(inputs=[input1, input2], outputs=[embedding])
    print([i.shape for i in model.weights])
    print(model.summary())
