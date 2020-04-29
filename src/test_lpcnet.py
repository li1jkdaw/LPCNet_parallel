#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Test an LPCNet model

import sys
import time
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import lpcnet
from ulaw import ulaw2lin, lin2ulaw

frame_size = lpcnet.frame_size
nb_features = lpcnet.nb_features
lpc_order = lpcnet.lpc_order


config = tf.ConfigProto()
set_session(tf.Session(config=config))

model, encoder, rnn_decoder = lpcnet.new_lpcnet_model(use_gpu=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                                metrics=['sparse_categorical_accuracy'])

feature_file = sys.argv[1]
out_file = sys.argv[2]
model_name = sys.argv[3]
nb_used_features = model.nb_used_features

print('Reading features from file %s...' % feature_file)
features = np.fromfile(feature_file, dtype='float32')
print('\nFeatures read.\n')

print('Processing features...')
features = np.reshape(features, (-1, nb_features))
feature_chunk_size = features.shape[0]
pcm_chunk_size = frame_size*feature_chunk_size
features = np.reshape(features, (1, feature_chunk_size, nb_features))
features[:,:,18:36] = 0
periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')
print('Data processing done!\n')

print('Loading model %s.\n' % model_name)
model.load_weights(model_name)

pcm = np.zeros((pcm_chunk_size, ))
pcm_input = np.zeros((1, 1, 3), dtype='int16')+128
state1 = np.zeros((1, model.rnn_units1), dtype='float32')
state2 = np.zeros((1, model.rnn_units2), dtype='float32')

mem = 0.0
coef = 0.85

if not out_file.endswith('.pcm'):
    print('\nOutput filename should end with .pcm!\n')
fout = open(out_file, 'wb')

print('Synthesis started.\n')
skip = lpc_order + 1
t = time.time()
enc_feat = encoder.predict([features[:, :, :nb_used_features], periods[:, :, :]])
for k in range(0, pcm_chunk_size):
    fr = k // frame_size
    if k < skip:
        continue

    a = features[0, fr, nb_features-lpc_order:]
    pred = -sum(a*pcm[k-1:k-lpc_order-1:-1])

    pcm_input[0, 0, 1] = lin2ulaw(pred)
    params, state1, state2 = rnn_decoder.predict([pcm_input, enc_feat[:, fr:fr+1, :], state1, state2])

#    Lower the temperature for voiced frames to reduce noisiness
    p = params * np.power(params, np.maximum(0, 1.5*features[0, fr, 37] - .5))
    p = p/(1e-18 + np.sum(p))
#    Cut off the tail of the remaining distribution
    p = np.maximum(p-0.002, 0).astype('float64')
    p = p/(1e-6 + np.sum(p))
    pcm_input[0, 0, 2] = np.argmax(np.random.multinomial(1, p[0,0,:], 1))

    pcm[k] = pred + ulaw2lin(pcm_input[0, 0, 2])
    pcm_input[0, 0, 0] = lin2ulaw(pcm[k])

    mem = coef*mem + pcm[k]
    np.array([np.round(mem)], dtype='int16').tofile(fout)

t = time.time() - t
print('Synthesis took %.2f seconds.' % t)
fout.close()