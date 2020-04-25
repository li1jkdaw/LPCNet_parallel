#!/bin/sh

gcc -Wall -W -O3 -g -I../include dump_data.c freq.c kiss_fft.c pitch.c celt_lpc.c common.c lpcnet_enc.c lpcnet_dec.c ceps_codebooks.c -o dump_data -lm
