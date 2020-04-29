import argparse
import random
import numpy as np
import time
import codecs
import os
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

import lpcnet
import loss


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--log_file', type=str, default='')
    args = parser.parse_args(argv)

    frames_for_separator = lpcnet.frames_for_separator
    nb_features = lpcnet.nb_features

    config = tf.ConfigProto()
    set_session(tf.Session(config=config))

    nb_rounds = 10
    nb_epochs = 25
    nb_records = 250
    batch_size = 128

    metrics = [loss.separator_loss]
    frame_processor = lpcnet.separator(training=True)
    frame_processor.compile(optimizer='adam', loss=loss.separator_loss, 
                            metrics=metrics)
    frame_processor.summary()
    nb_used_features = frame_processor.nb_used_features

    data_dir = args.data_dir
    log_file = args.log_file

    pcms_path = os.path.join(data_dir, 'pcms')
    pcm_dir = os.path.join(data_dir, 'tmp')
    feat_dir = os.path.join(data_dir, 'feats')
    pcm_files = [f for f in os.listdir(pcms_path) 
                 if os.path.isfile(os.path.join(pcms_path, f))]
    pcm_files = [f for f in pcm_files if f.endswith('.pcm')]
    print('%d pcm files found in %s' % (len(pcm_files), pcms_path))

    for k in range(nb_rounds):
        print('Starting round %d\n' % (k + 1))

        if k > 0:
            command = 'rm %s/*' % pcm_dir
            print('Executing %s...' % command)
            os.system(command)
            command = 'rm %s/*' % feat_dir
            print('Executing %s...' % command)
            os.system(command)

        random.shuffle(pcm_files)
        for pcm_file in pcm_files[:nb_records]:
            command = 'cp %s %s/' % (os.path.join(pcms_path, pcm_file),
                                     os.path.join(pcm_dir))
            print('Executing %s...' % command)
            os.system(command)
        
        print('Running preparing_training_dataset.sh...')
        t = time.time()
        os.system('bash prepare_training_dataset.sh %s %s' % (pcm_dir, feat_dir))
        t = time.time() - t
        print('Dataset preparation took %.1f minutes.' % (t/60.0))

        feature_file = os.path.join(feat_dir, 'features.f32')
        print('Reading audio features from file %s...' % feature_file)
        features = np.fromfile(feature_file, dtype=np.float32)
        loss_file = os.path.join(feat_dir, 'error.loss')
        print('Reading loss from file %s...' % loss_file)
        error = np.fromfile(loss_file, dtype=np.float32)
        print('Data read.')

        nb_batches = len(features)//(nb_features*frames_for_separator)
        print('Total number of training batches is %d.' % nb_batches)

        print('Processing acoustic features...')

        features = features[:nb_batches*frames_for_separator*nb_features]
        features = np.reshape(features, (nb_batches, frames_for_separator, nb_features))
        features = features[:, :, :nb_used_features]
        features[:,:,18:36] = 0
        fpad1 = np.concatenate([features[:1, :2, :], features[:-1, -2:, :]], axis=0)
        fpad2 = np.concatenate([features[1:, :2, :], features[:1, -2:, :]], axis=0)
        features = np.concatenate([fpad1, features, fpad2], axis=1)

        periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')

        error = error[:2*nb_batches*frames_for_separator]
        error = np.reshape(error, (nb_batches, frames_for_separator, 2))

        print('Data processing done!\n')

        if k > 0:
            print('Loading model...')
            frame_processor.load_weights('separator.h5')

        checkpoint = ModelCheckpoint('separator.h5')
        logger = lpcnet.log_separator(log_file)
        with codecs.open(log_file, 'a', 'utf-8') as f:
            f.write('\nRound %d:\n' % (k+1))
        lr = 0.001

        print('Training started.\n')
        frame_processor.compile(optimizer=Adam(lr, amsgrad=True, decay=0.0), 
                                loss=loss.separator_loss)
        frame_processor.fit([features, periods], error, 
                            batch_size=batch_size, epochs=nb_epochs, 
                            validation_split=0.0, callbacks=[checkpoint, logger])
        
        print('Training finished.\n')

    command = 'rm %s/*' % pcm_dir
    print('Executing %s...' % command)
    os.system(command)
    command = 'rm %s/*' % feat_dir
    print('Executing %s...' % command)
    os.system(command)


if __name__ == '__main__':
	main()
