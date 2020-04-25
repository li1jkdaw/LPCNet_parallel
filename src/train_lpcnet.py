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

# Train an LPCNet model

import sys
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

import lpcnet


feature_chunk_size = lpcnet.feature_chunk_size
frame_size = lpcnet.frame_size
pcm_chunk_size = frame_size*feature_chunk_size
nb_features = lpcnet.nb_features

config = tf.ConfigProto()
set_session(tf.Session(config=config))

nb_epochs = 10000
batch_size = 256

model, _, _ = lpcnet.new_lpcnet_model(training=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                                metrics=['sparse_categorical_accuracy'])
model.summary()

feature_file = sys.argv[1]
pcm_file = sys.argv[2]
log_file = sys.argv[3]
nb_used_features = model.nb_used_features

# Reading data from file
print('Reading data from file %s...' % pcm_file)
data = np.fromfile(pcm_file, dtype='uint8')
print('\nData read.\n')

nb_frames = len(data)//(4*pcm_chunk_size)
print('Total number of training batches is %d.\n' % nb_frames)

print('Processing data...')
data = data[:nb_frames*4*pcm_chunk_size]
sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))
del data

print('Rearranging data...')
in_data = np.concatenate([sig, pred, in_exc], axis=-1)
del sig
del pred
del in_exc

# Reading features from data
print('Reading features from file %s...' % feature_file)
features = np.fromfile(feature_file, dtype='float32')
print('\nFeatures read.\n')

print('Processing acoustic features...')
features = features[:nb_frames*feature_chunk_size*nb_features]
features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :nb_used_features]
features[:,:,18:36] = 0
fpad1 = np.concatenate([features[:1, :2, :], features[:-1, -2:, :]], axis=0)
fpad2 = np.concatenate([features[1:, :2, :], features[:1, -2:, :]], axis=0)
features = np.concatenate([fpad1, features, fpad2], axis=1)
periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')
print('Data processing done!\n')

# Callbacks
checkpoint = ModelCheckpoint('lpcnet_{epoch:02d}.h5')
sparsify = lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2), log_file)

lr = 0.001
decay = 5e-5

print('Training started.\n')
model.compile(optimizer=Adam(lr, amsgrad=True, decay=decay), 
              loss='sparse_categorical_crossentropy')
model.save_weights('lpcnet_00.h5')
model.fit([in_data, features, periods], out_exc, 
          batch_size=batch_size, epochs=nb_epochs, validation_split=0.0,
          callbacks=[checkpoint, sparsify])