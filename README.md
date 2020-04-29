# Parallel LPCNet

Low complexity implementation of the WaveRNN-based LPCNet algorithm is described in:

- J.-M. Valin, J. Skoglund, [A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet](https://jmvalin.ca/papers/lpcnet_codec.pdf), *Submitted for INTERSPEECH 2019*.
- J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Proc. International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, arXiv:1810.11846, 2019.

The code for parallel LPCNet is based on original LPCNet code from https://github.com/mozilla/LPCNet.

Please note that it is not a real multhithread implementation of parallel LPCNet. It is just the simulation of it in order to get audio records which sound as if they were synthesized in a parallel manner.

# Introduction

You can build the code using:

```
mkdir build
cd build
cmake ..
make
```

Use 
```
cd src
bash compile.sh
``` 
to generate *dump\_data* executable.

# Training a new LPCNet model

1. Set up a Keras system with GPU.

2. Generate training data for a signle pcm file:
   ```
   src/dump_data -train input.pcm data/features_train.i8 data/data_train.f32
   ```
   or for multiple pcm files:
   ```
   bash get_training_data.sh /path/to/pcm/directory /path/to/output/data/directory
   ```
   where input.pcm contains 16 kHz 16-bit raw PCM audio (no header) and the other files are output files.

3. Now that you have your files, train with:
   ```
   python src/train_lpcnet.py data/features_train.f32 data/data_train.i8 logs/log.txt
   ```
   and it will generate an lpcnet_*.h5 file for each epoch. If it stops with a
   "Failed to allocate RNN reserve space" message try reducing the *batch\_size* variable in train_lpcnet.py.

# Training a new separator model (neural net for splitting frames detection)

1. You may use default model/separator.h5 trained on LJSpeech. Its inputs are 18 BFCC features for the signal with preemphasis (coefficient=0.85). So, if input features for LPCNet are different or if the default model doens't work well for your data, a new separator model should be trained.

2. To train your own model, run
   ```
   cd src
   python train_separator.py --data_dir /path/to/data/directory --log_file /path/to/log/file
   ```
   Your data directory should consist of three subfolders: "pcms" with pcm training files and empty "tmp" and "feats" folders. Also, you should have your LPCNet vocoder trained and C code built (see next section, part 2) since separator model is trained on records synthesized with LPCNet for which splitting frames allocated at random.

# Testing a model

1. Create a feature file out of test.pcm:
   ```
   src/dump_data -test test.pcm data/features_test.f32
   ```

2. For a new model first extract the model files nnet_data.h and nnet_data.c
   ```
   cd src && python dump_lpcnet.py /path/to/your/lpcnet/model.h5 /path/to/your/separator/model.h5 && cd ..
   ```
   and then build C code. If you don't have separator model trained (or if you are not even going to train it and use for synthesis), please use the default model/separator.h5. 

3. Run ./LPCNet to synthesize speech:
   ```
   ./LPCNet -synthesis_std data/features_test.f32 audio/generated_audio.pcm
   ```
   Instead of the flag -synthesis_std you may use -synthesis_rule, -synthesis_net, -synthesis_smooth or -synthesis_shift:

   -synthesis_std means that you run original (non-parallel) LPCNet model;

   -synthesis_rule means that you run parallel LPCNet for which splitting frames are detected using a simple energy-based rule (EB-splitting);

   -synthesis_net means that you run parallel LPCNet for which splitting frames are detected using a neural network separator.h5 (NN-splitting);

   -synthesis_smooth means that you run parallel LPCNet with overlapping for which each 10-th frame is a splitting frame and two waveforms in each overlapping frame are integrated using linear cross-fading without shift (XF w/o shift);

   -synthesis_shift means that you run parallel LPCNet with overlapping for which each 10-th frame is a splitting frame and two waveforms in each overlapping region are integrated using linear cross-fading with shift (XF with shift);

  <audio controls="controls">
    <source src="mp3/.mp3" type="audio/mp3">
  </audio>
