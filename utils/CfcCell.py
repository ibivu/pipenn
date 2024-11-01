import datetime, time, os, sys, inspect
#import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Reshape, Input, concatenate, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Activation, \
                                           BatchNormalization, Conv2DTranspose, ZeroPadding2D, Dropout, UpSampling2D, UpSampling1D, \
                                           LSTM, GRU, TimeDistributed, Bidirectional, Dense, Add, ConvLSTM2D, Flatten, AveragePooling1D, \
                                           ReLU, LeakyReLU, PReLU, ELU

def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666*x)


class CfcCell(tf.keras.layers.Layer):
    def __init__(self, units, hparams, **kwargs):
        super(CfcCell, self).__init__(**kwargs)
        self.units      = units   #Number of RNN units
        self.state_size = units
        self.hparams    = hparams
        self.no_gate    = False
    
    def build(self, input_shape):
        if self.hparams.get("backbone_activation") == "silu":
            backbone_activation = tf.nn.silu
        elif self.hparams.get("backbone_activation") == "relu":
            backbone_activation = tf.nn.relu
        elif self.hparams.get("backbone_activation") == "tanh":
            backbone_activation = tf.nn.tanh
        elif self.hparams.get("backbone_activation") == "gelu":
            backbone_activation = tf.nn.gelu
        elif self.hparams.get("backbone_activation") == "lecun":
            backbone_activation = lecun_tanh
        elif self.hparams.get("backbone_activation") == "softplus":
            backbone_activation = tf.nn.softplus
        else:
            raise ValueError("Unknown backbone activation")
        
        self.no_gate = self.hparams["no_gate"]

        self.backbone = []
        for i in range(self.hparams["backbone_layers"]):
            self.backbone.append(
                tf.keras.layers.Dense(
                    self.hparams["backbone_units"],
                    backbone_activation,
                    kernel_regularizer=tf.keras.regularizers.L2(
                        self.hparams["weight_decay"]
                    ),
                )
            )
            self.backbone.append(tf.keras.layers.Dropout(self.hparams["backbone_dr"]))
        self.backbone = tf.keras.models.Sequential(self.backbone)

        self.ff1 = tf.keras.layers.Dense(
                self.units,
                lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
        self.ff2 = tf.keras.layers.Dense(
            self.units,
            lecun_tanh,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
        )
        self.time_a = tf.keras.layers.Dense(
            self.units,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
        ) 
        self.time_b = tf.keras.layers.Dense(
            self.units,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
        )
    
    def call(self, inputs, state):##state is create automatically and passed as arguments for the next point in the sequennce 
        hidden_state = state[0]
        t = 1.0
        x = tf.keras.layers.Concatenate()([inputs, hidden_state])
       
        x = self.backbone(x)
        ff1 = self.ff1(x)
        ff2 = self.ff2(x)
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = tf.nn.sigmoid(-t_a * t + t_b)
        if self.no_gate:
            new_hidden = ff1 + t_interp * ff2
        else:
            new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]
    
    def get_config(self):
        config = super(CfcCell, self).get_config()
        config.update({
            'units': self.units,
            'hparams': self.hparams,
            # You might need to serialize 'hparams' if it's not a primitive type
        })
        return config