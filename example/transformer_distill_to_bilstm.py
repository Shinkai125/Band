"""
@file: transformer_distill_to_bilstm.py
@time: 2020-09-21 20:48:22
"""

from tensorflow import keras
from tensorflow.keras import layers

from model.distiller import Distiller
from model.transformer import Transformer

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

# Create the teacher
transformer = Transformer(vocab_size=vocab_size, maxlen=maxlen, feed_forward_dim=32)
sequence_output, pooled_output = transformer.outputs
x = layers.Dropout(0.1)(pooled_output)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="sigmoid")(x)
teacher = keras.Model(inputs=transformer.inputs, outputs=outputs, name="teacher")
print(teacher.summary())

# Create the student
inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(vocab_size + 1, 64)(inputs)
x = layers.Bidirectional(layers.LSTM(16))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
student = keras.Model(inputs=inputs, outputs=outputs, name="student")
print(student.summary())

print("Train teacher as usual")
teacher.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = teacher.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
)

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
print("Distill teacher to student")
distiller.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val))
