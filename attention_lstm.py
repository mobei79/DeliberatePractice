#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/19 下午2:45
# @Author  : xiao.lin
# @Site    : 
# @File    : attention_lstm.py
# @Software: PyCharm Community Edition
from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper,GRU,concatenate
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.regularizers import l2


class AttentionGRU(GRU):

  def __init__(self, atten_states, states_len, L2Strength, **kwargs):
    '''
    :param atten_states: previous states for attention
    :param states_len: length of state
    :param L2Strength: for regularization
    :param kwargs: for GRU
    '''
    self.p_states = atten_states
    self.states_len = states_len
    self.size = kwargs['units']
    self.L2Strength = L2Strength
    super(AttentionGRU, self).__init__(**kwargs)

  def build(self, input_shape):
    if hasattr(self.p_states, '_keras_shape'):
        attention_dim = self.p_states._keras_shape[1]
    else:
        raise Exception('Layer could not be build: No information about expected input shape.')

    self.W1 = self.add_weight(name='{}_W_1'.format(self.name),
                              shape = (self.units, 1,),
                              initializer = 'glorot_uniform',
                              regularizer=l2(self.L2Strength),
                              trainable=True
                              )
    self.b1 = self.add_weight(name='{}_b_1'.format(self.name),
                              shape=(1,),
                              initializer = 'zero',
                              regularizer=l2(self.L2Strength),
                              trainable= True)
    self.W2 = self.add_weight(name='{}_W_2'.format(self.name),
                              shape=(self.units, self.units),
                              initializer='glorot_uniform',
                              regularizer=l2(self.L2Strength),
                              trainable=True)
    self.b2 = self.add_weight(name='{}_b_2'.format(self.name),
                              shape=(self.units,),
                              initializer='zero',
                              regularizer=l2(self.L2Strength),
                              trainable=True)
    self.W3 = self.add_weight(name='{}_W_3'.format(self.name),
                              shape=(self.units, self.units),
                              initializer='glorot_uniform',
                              regularizer=l2(self.L2Strength),
                              trainable=True)
    self.b3 = self.add_weight(name='{}_b_3'.format(self.name),
                              shape=(self.units,),
                              initializer='zero',
                              regularizer=l2(self.L2Strength),
                              trainable=True)
    self.U_m = self.add_weight((attention_dim, self.units),
                               initializer='glorot_uniform',
                               name='{}_U_m'.format(self.name),
                               regularizer=l2(self.L2Strength),
                               trainable=True
                               )
    self.b_m = K.zeros((self.units,), name='{}_b_m'.format(self.name))

    super(AttentionGRU, self).build(input_shape)

  def step(self, inputs, states):
    h, _= super(AttentionGRU, self).step(inputs, states)
    attention = states[-1]

    m = K.tanh((K.dot(h, self.W1)+ self.b1)+(K.dot(attention, self.W3)+ self.b3))
    s = K.sigmoid(K.dot(m, self.W2) + self.b2)
    h = h * s

    return h, [h]
    # print(h)
    # alfa = K.repeat(h, self.states_len) # alfa = [batch_size, states_len, units]
    # alfa = K.concatenate([self.p_states, alfa], axis = 2) # alfa = [batch_size, states_len, 2*units]
    # scores = K.tanh(K.dot(alfa, self.W1) + self.b1) # scores = [batch_size, states_len, 1]
    # scores = K.softmax(scores)
    # scores = K.reshape(scores, (-1, 1, self.states_len)) # scores = [batch_size, 1, states_len]
    # attn = K.batch_dot(scores, self.p_states) # attn = [batch_size, 1, units]
    # attn = K.reshape(attn, (-1, self.units))  # attn = [batch_size, units]
    #
    # h = concatenate([h, attn]) # h = [batch_size, 2*units]
    # h = K.dot(h, self.W2) + self.b2 # h = [batch_size, units]
    # return h, [h]x

  def get_constants(self,x, training=None):
      constants = super(AttentionGRU, self).get_constants(x)
      constants.append(K.dot(self.p_states, self.U_m) + self.b_m)
      return constants

  def compute_output_shape(self, input_shape):
      return input_shape[0],self.units #input_shape[1],


class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        self.init = initializers.get('glorot_uniform')
        self.output_dim = output_dim
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.add_weight((self.output_dim, self.output_dim,),
                                 initializer=self.init,
                                 name='{}_U_a'.format(self.name))

        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.add_weight((attention_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.add_weight((self.output_dim, 1),
                                       initializer=self.init,
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

        # if self.initial_weights is not None:
        #     self.set_weights(self.initial_weights)
        #     del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[-1]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
        # which I think might have been caused by the exponential function -> gradients blow up)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

class AttentionLSTMWrapper(Wrapper):
    def __init__(self, layer, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        self.init = initializers.get('glorot_uniform')
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build()

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.add_weight((self.layer.units, self.layer.units),initializer=self.init, name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.layer.units,), name='{}_b_a'.format(self.name))

        self.U_m = self.add_weight((attention_dim, self.layer.units),initializer=self.init, name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.layer.units,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.add_weight((self.layer.units, 1),initializer=self.init, name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.add_weight((self.layer.units, self.layer.units),initializer=self.init, name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.layer.units,), name='{}_b_s'.format(self.name))

        self.layer.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.recurrent_activation(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output


class Attention_layer(Layer):
    """ 
        Attention operation, with a context/query vector, for temporal data. 
        Supports Masking. 
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf] 
        "Hierarchical Attention Networks for Document Classification" 
        by using a context vector to assist the attention 
        # Input shape 
            3D tensor with shape: `(samples, steps, features)`. 
        # Output shape 
            2D tensor with shape: `(samples, features)`. 
        :param kwargs: 
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True. 
        The dimensions are inferred based on the output shape of the RNN. 
        Example: 
            model.add(LSTM(64, return_sequences=True)) 
            model.add(AttentionWithContext()) 
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        #[batch,step,num_units]
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

            # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
