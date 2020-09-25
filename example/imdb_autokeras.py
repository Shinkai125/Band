"""
@file: imdb_autokeras.py
@time: 2020-09-23 11:04:04
"""
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset.imdb_text import imdb_to_text

# Prepare the data.

(x_train, y_train), (x_test, y_test) = imdb_to_text()
print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # <START> this film was just brilliant casting <UNK>

# Initialize the TextClassifier
clf = ak.TextClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train, epochs=2)
# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)))

# Export as a Keras Model
model = clf.export_model()
print(model.summary())

# print model as image
tf.keras.utils.plot_model(
    model, show_shapes=True, expand_nested=True, to_file="name.png"
)

try:
    model.save("model_autokeras", save_format="tf")
except:
    model.save("model_autokeras.h5")


loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
print(
    "Eval Accuracy: {accuracy}".format(
        accuracy=loaded_model.evaluate(x_test, y_test)
    )
)

predicted_y = loaded_model.predict(x_test)
print(predicted_y.shape)
