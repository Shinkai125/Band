from tensorflow import keras
from tensorflow.keras import layers

from model.transformer import Transformer

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=vocab_size
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

transformer = Transformer(
    vocab_size=vocab_size, maxlen=maxlen, feed_forward_dim=32
)
sequence_output, pooled_output = transformer.outputs
x = layers.Dropout(0.1)(pooled_output)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=transformer.inputs, outputs=outputs)

print(model.summary(line_length=150))

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
)
