"""
@file: bert.py
@time: 2020-09-23 15:27:12
"""
import codecs
import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers.TokenAndPositionEmbedding import BERTEmbedding
from layers.TransformerBlock import TransformerBlock


def BERT(
        vocab_size=2000,
        maxlen=100,
        embed_dim=256,
        num_heads=2,
        feed_forward_dim=256,
        num_layers=1,
):
    """
    :param num_layers: Transformer Block number
    :param vocab_size: Vocabulary size
    :param maxlen: Max sequence size
    :param embed_dim: Embedding size for each token
    :param num_heads: Number of attention heads
    :param feed_forward_dim: Hidden layer size in feed forward network inside transformer
    :return: tf.keras.Model
    """
    token_input = layers.Input(
        shape=(maxlen,), dtype=tf.int32, name="Input-Token"
    )
    segment_input = layers.Input(
        shape=(maxlen,), dtype=tf.int32, name="Input-Segment"
    )

    embedding = BERTEmbedding(maxlen, vocab_size, embed_dim)(
        [token_input, segment_input]
    )

    sequence_output = None
    for i in range(num_layers):
        if i == 0:
            sequence_output = TransformerBlock(
                embed_dim,
                num_heads,
                feed_forward_dim,
                name="TransformerBlock-%s" % i,
            )(embedding)
        else:
            sequence_output = TransformerBlock(
                embed_dim,
                num_heads,
                feed_forward_dim,
                name="TransformerBlock-%s" % i,
            )(sequence_output)

    pooled_output = layers.GlobalAveragePooling1D(name="pooled_output")(
        sequence_output
    )
    model = keras.Model(
        inputs=[token_input, segment_input],
        outputs=[sequence_output, pooled_output],
    )
    return model


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_model_from_pretrained_model(model, pretrain_dir, config):
    model_checkpoint_path = os.path.join(pretrain_dir, "bert_model.ckpt")
    loader = checkpoint_loader(model_checkpoint_path)

    model.get_layer(name="bert_embedding").set_weights(
        [
            loader("bert/embeddings/word_embeddings"),
            loader("bert/embeddings/token_type_embeddings"),
            loader("bert/embeddings/position_embeddings"),
            loader("bert/embeddings/LayerNorm/gamma"),
            loader("bert/embeddings/LayerNorm/beta"),
        ]
    )

    for i in range(config["num_hidden_layers"]):
        model.get_layer(name="TransformerBlock-%d" % i).set_weights(
            [
                loader(
                    "bert/encoder/layer_%d/attention/self/query/kernel" % i
                ),
                loader("bert/encoder/layer_%d/attention/self/query/bias" % i),
                loader("bert/encoder/layer_%d/attention/self/key/kernel" % i),
                loader("bert/encoder/layer_%d/attention/self/key/bias" % i),
                loader(
                    "bert/encoder/layer_%d/attention/self/value/kernel" % i
                ),
                loader("bert/encoder/layer_%d/attention/self/value/bias" % i),
                loader(
                    "bert/encoder/layer_%d/attention/output/dense/kernel" % i
                ),
                loader(
                    "bert/encoder/layer_%d/attention/output/dense/bias" % i
                ),
                loader("bert/encoder/layer_%d/intermediate/dense/kernel" % i),
                loader("bert/encoder/layer_%d/intermediate/dense/bias" % i),
                loader("bert/encoder/layer_%d/output/dense/kernel" % i),
                loader("bert/encoder/layer_%d/output/dense/bias" % i),
                loader(
                    "bert/encoder/layer_%d/attention/output/LayerNorm/gamma"
                    % i
                ),
                loader(
                    "bert/encoder/layer_%d/attention/output/LayerNorm/beta" % i
                ),
                loader("bert/encoder/layer_%d/output/LayerNorm/gamma" % i),
                loader("bert/encoder/layer_%d/output/LayerNorm/beta" % i),
            ]
        )

    return model


class BERT_Model(object):
    def __init__(
            self,
            pretrained_model_dir,
            config_file="bert_config.json",
            vocab_file="vocab.txt",
    ):
        self.pretrained_model_dir = pretrained_model_dir
        self.config = self.load_config(pretrained_model_dir, config_file)
        self.vocab = self.load_vocabulary(vocab_file)
        self.bert = self.build_model(self.config)

    @staticmethod
    def load_config(pretrained_model_dir, config_file):
        with open(
                os.path.join(pretrained_model_dir, config_file), "r"
        ) as load_f:
            config = json.load(load_f)
        return config

    @staticmethod
    def load_vocabulary(vocab_path):
        token_dict = {}
        with codecs.open(vocab_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict

    @staticmethod
    def build_model(config):
        model = BERT(
            maxlen=512,
            vocab_size=config["vocab_size"],
            embed_dim=config["hidden_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            feed_forward_dim=config["intermediate_size"],
        )
        return model

    def from_pretrained(self):
        model = load_model_from_pretrained_model(
            model=self.bert,
            pretrain_dir=self.pretrained_model_dir,
            config=self.config,
        )
        return model


# if __name__ == '__main__':
#     model = BERT_Model(pretrained_model_dir='/home/lic2020/models/chinese_L-12_H-768_A-12').from_pretrained()
#     print(model.summary(line_length=150))

if __name__ == "__main__":
    model = BERT(
        maxlen=512,
        vocab_size=21128,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        feed_forward_dim=3072,
    )
    model.summary(line_length=150)
