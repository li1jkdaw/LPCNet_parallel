/* Copyright (c) 2018 Mozilla */
/* modified by Vadim Popov */
/*
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
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include <time.h>
#include "arch.h"
#include "lpcnet.h"
#include "freq.h"
#include "frame_analysis.h"

#define MAX_RESET_FRAMES 1000
#define MIN_FRAMES_WO_RESET 10
#define NUM_CONSEC_4NET 3
#define RESET_PROBABILITY 0.2
#define RESET_THRESHOLD 0.95
#define HISS_THRESHOLD 4.3
#define MAX_SHIFT 80
#define SIMILARITY_WINDOW 80
#define SMOOTH_FACTOR 3.0

#define MODE_ENCODE 0
#define MODE_DECODE 1
#define MODE_SYNTHESIS_STD 2
#define MODE_SYNTHESIS_RULE 3
#define MODE_SYNTHESIS_NET 4
#define MODE_SYNTHESIS_SMOOTH 5
#define MODE_SYNTHESIS_SHIFT 6
#define MODE_RESET 7
#define MODE_SYNTHESIS_FAKE 8
#define MODE_ERROR 9

void print_usage(int n)
{
    fprintf(stderr, "Error: %d is incorrect number of arguments.\n", n-1); 
    fprintf(stderr, "Usage: LPCNet -encode <input.pcm> <compressed.lpcnet>\n");
    fprintf(stderr, "       LPCNet -decode <compressed.lpcnet> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -synthesis_std <features.f32> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -synthesis_rule <features.f32> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -synthesis_net <features.f32> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -synthesis_smooth <features.f32> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -synthesis_shift <features.f32> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -reset <features.f32> <resets.msk>\n");
    fprintf(stderr, "       LPCNet -synthesis_fake <features.f32> <resets.msk> <output.pcm>\n");
    fprintf(stderr, "       LPCNet -error <features_real.f32> <features_fake.f32> <resets.msk> <error_out.loss>\n");
    return;
}

void insert_silent_frames(FILE *f)
{
    int i;
    short pcm[LPCNET_FRAME_SIZE];
    for (i = 0; i < LPCNET_FRAME_SIZE; i++)
        pcm[i] = 0;
    for (i = 0; i < 10; i++)
        fwrite(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, f);
    return;
}

void remove_hiss(float *features)
{
    float features_modified[NB_BANDS];
    int i;
    idct(features_modified, features);
    for (i = 14; i < NB_BANDS; i++)
    {
        if (features_modified[i] > HISS_THRESHOLD)
            features_modified[i] = HISS_THRESHOLD;
    }
    dct(features, features_modified);
    return;
}


int main(int argc, char **argv) {
    int mode;
    int r;
    FILE *fin1, *fin2, *fin3, *fout;
    if (argc < 2)
    {
        print_usage(argc);
        return 0;
    }
    if (strcmp(argv[1], "-encode") == 0) mode = MODE_ENCODE;
    else if (strcmp(argv[1], "-decode") == 0) mode = MODE_DECODE;
    else if (strcmp(argv[1], "-synthesis_std") == 0) mode = MODE_SYNTHESIS_STD;
    else if (strcmp(argv[1], "-synthesis_rule") == 0) mode = MODE_SYNTHESIS_RULE;
    else if (strcmp(argv[1], "-synthesis_net") == 0) mode = MODE_SYNTHESIS_NET;
    else if (strcmp(argv[1], "-synthesis_smooth") == 0) mode = MODE_SYNTHESIS_SMOOTH;
    else if (strcmp(argv[1], "-synthesis_shift") == 0) mode = MODE_SYNTHESIS_SHIFT;
    else if (strcmp(argv[1], "-reset") == 0) mode = MODE_RESET;
    else if (strcmp(argv[1], "-synthesis_fake") == 0) mode = MODE_SYNTHESIS_FAKE;
    else if (strcmp(argv[1], "-error") == 0) mode = MODE_ERROR;
    else 
    {
        print_usage(argc);
        exit(1);
    }
    if ((mode < MODE_SYNTHESIS_FAKE && argc != 4) || (mode == MODE_SYNTHESIS_FAKE && argc != 5) || (mode == MODE_ERROR && argc != 6))
    {
        print_usage(argc);
        return 0;
    }
    fin1 = fopen(argv[2], "rb");
    if (fin1 == NULL) 
    {
	    fprintf(stderr, "Can't open %s\n", argv[2]);
	    exit(1);
    }
    if (mode == MODE_SYNTHESIS_FAKE || mode == MODE_ERROR)
    {
        fin2 = fopen(argv[3], "rb");
        if (fin2 == NULL)
        {
	        fprintf(stderr, "Can't open %s\n", argv[3]);
	        exit(1);
        }
    }
    if (mode != MODE_ERROR)
    {
        fout = fopen(argv[argc-1], "wb");
        if (fout == NULL) 
        {
	        fprintf(stderr, "Can't open %s\n", argv[argc-1]);
	        exit(1);
        }
    }
    else
    {
        fin3 = fopen(argv[4], "rb");
        if (fin3 == NULL) 
        {
	        fprintf(stderr, "Can't open %s\n", argv[4]);
	        exit(1);
        }
        fout = fopen(argv[argc-1], "a");
        if (fout == NULL) 
        {
	        fprintf(stderr, "Can't open %s\n", argv[argc-1]);
	        exit(1);
        }
    }

    if (mode == MODE_ENCODE) 
    {
        LPCNetEncState* net;
        net = lpcnet_encoder_create();
		while (1)
        {
            unsigned char buf[LPCNET_COMPRESSED_SIZE];
            short pcm[LPCNET_PACKET_SAMPLES];
            r = fread(pcm, sizeof(pcm[0]), LPCNET_PACKET_SAMPLES, fin1);
            if (feof(fin1)) break;
            lpcnet_encode(net, pcm, buf);
            fwrite(buf, 1, LPCNET_COMPRESSED_SIZE, fout);
        }
        lpcnet_encoder_destroy(net);
    } 
    else if (mode == MODE_DECODE) 
    {
        LPCNetDecState *net;
        net = lpcnet_decoder_create();
        while (1) 
        {
            unsigned char buf[LPCNET_COMPRESSED_SIZE];
            short pcm[LPCNET_PACKET_SAMPLES];
            r = fread(buf, sizeof(buf[0]), LPCNET_COMPRESSED_SIZE, fin1);
            if (feof(fin1)) break;
            lpcnet_decode(net, buf, pcm);
            fwrite(pcm, sizeof(pcm[0]), LPCNET_PACKET_SAMPLES, fout);
        }
        lpcnet_decoder_destroy(net);
    }
    else if (mode == MODE_SYNTHESIS_STD) 
    {
        LPCNetState *net;
        net = lpcnet_create();
        printf("Synthesis...\n");
        while (1)
        {
            float in_features[NB_TOTAL_FEATURES];
            float features[NB_FEATURES];
            short pcm[LPCNET_FRAME_SIZE];
            r = fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin1);
            if (feof(fin1)) break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
//            remove_hiss(features);
            srand(net->frame_count + 23);
            lpcnet_synthesize(net, features, pcm, LPCNET_FRAME_SIZE, 0);
            fwrite(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, fout);
        }
//        insert_silent_frames(fout);
        lpcnet_destroy(net);
    }
    else if (mode == MODE_SYNTHESIS_RULE) 
    {
        LPCNetState *net;
        net = lpcnet_create();
        int *reset_frames;
        reset_frames = (int*)malloc(MAX_RESET_FRAMES*sizeof(int));
        int num_reset_frames;
        int reset_count = 0;
        num_reset_frames = get_reset_frames(argv[2], reset_frames);
        printf("Synthesis...\n");
        while (1) 
        {
            float in_features[NB_TOTAL_FEATURES];
            float features[NB_FEATURES];
            short pcm[LPCNET_FRAME_SIZE];
            int is_reset_frame = 0;
            if (reset_count < num_reset_frames && reset_frames[reset_count] == net->frame_count)
            {
                is_reset_frame = 1;
                reset_count++;
            }
            r = fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin1);
            if (feof(fin1)) break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
//            remove_hiss(features);
            srand(net->frame_count + 23);
            lpcnet_synthesize(net, features, pcm, LPCNET_FRAME_SIZE, is_reset_frame);
            fwrite(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, fout);
        }
//        insert_silent_frames(fout);
        printf("Number of reset frames is %d\n", num_reset_frames);
        lpcnet_destroy(net);
        free(reset_frames);
    }
    else if (mode == MODE_SYNTHESIS_NET) 
    {
        LPCNetState *net;
        net = lpcnet_create();
        int last_reset_frame = -1000;
        int num_consec_detected = 0;
        int num_resets = 0;
        printf("Synthesis...\n");
        while (1)
        {
            float in_features[NB_TOTAL_FEATURES];
            float features[NB_FEATURES];
            short pcm[LPCNET_FRAME_SIZE];
            int is_reset_frame = 0;
            float prob;
            r = fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin1);
            if (feof(fin1)) break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
//            remove_hiss(features);
            prob = calculate_reset_probs(net, features);
            if (net->frame_count >= 2*MIN_FRAMES_WO_RESET && 
                net->frame_count - last_reset_frame > 2*MIN_FRAMES_WO_RESET && 
                prob > RESET_THRESHOLD)
            {
                num_consec_detected += 1;
                if (num_consec_detected == NUM_CONSEC_4NET)
                {
                    is_reset_frame = 1;
                    last_reset_frame = net->frame_count;
                    num_resets++;
                }
            }
            else
                num_consec_detected = 0;
            srand(net->frame_count + 23);
            lpcnet_synthesize(net, features, pcm, LPCNET_FRAME_SIZE, is_reset_frame);
            fwrite(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, fout);
        }
//        insert_silent_frames(fout);
        printf("Number of reset frames is %d\n", num_resets);
        lpcnet_destroy(net);
    }
    else if (mode == MODE_SYNTHESIS_SMOOTH) 
    {
        LPCNetState *net1;
        net1 = lpcnet_create();
        LPCNetState *net2;
        net2 = lpcnet_create();
        int reset_count = 0;
        printf("Synthesis...\n");
        while (1)
        {
            float in_features[NB_TOTAL_FEATURES];
            float features[NB_FEATURES];
            short pcm1[LPCNET_FRAME_SIZE];
            short pcm2[LPCNET_FRAME_SIZE];
            int i, is_reset_frame = 0;
            float smoothed_pcm;
            if (net1->frame_count > 0 && net1->frame_count % MIN_FRAMES_WO_RESET == 0)
            {
                is_reset_frame = 1;
                reset_count++;
            }

            r = fread(in_features, sizeof(float), NB_TOTAL_FEATURES, fin1);
            if (feof(fin1)) break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
//            remove_hiss(features);
            if (is_reset_frame)
            {
                lpcnet_copy(net2, net1);
                srand(net2->frame_count + 23);
                lpcnet_synthesize(net2, features, pcm2, LPCNET_FRAME_SIZE, 0);
            }
            srand(net1->frame_count + 23);
            lpcnet_synthesize(net1, features, pcm1, LPCNET_FRAME_SIZE, is_reset_frame);
            if (is_reset_frame)
            {
                for (i = 0; i < FRAME_SIZE; i++)
                {
                    smoothed_pcm = ((FRAME_SIZE - i)*pcm2[i] + i*pcm1[i])/FRAME_SIZE;
                    pcm1[i] = (short)smoothed_pcm;
                }
            }
            fwrite(pcm1, sizeof(short), LPCNET_FRAME_SIZE, fout);
        }
//        insert_silent_frames(fout);
        printf("Number of reset frames is %d\n", reset_count);
        lpcnet_destroy(net1);
        lpcnet_destroy(net2);
    }
    else if (mode == MODE_SYNTHESIS_SHIFT) 
    {
        LPCNetState *net1;
        net1 = lpcnet_create();
        LPCNetState *net2;
        net2 = lpcnet_create();
        int reset_count = 0;
        printf("Synthesis...\n");
        while (1)
        {
            float in_features[NB_TOTAL_FEATURES];
            float features[NB_FEATURES];
            short pcm1[LPCNET_FRAME_SIZE];
            short pcm2[LPCNET_FRAME_SIZE];
            int i, j;
            int is_reset_frame = 0;
            int is_after_reset = 0;
            int shift = 0;
            
            float smoothed_pcm;
            if (net1->frame_count > 0 && net1->frame_count % MIN_FRAMES_WO_RESET == 0)
            {
                is_reset_frame = 1;
                reset_count++;
            }
            else if (net1->frame_count > 1  && net1->frame_count % MIN_FRAMES_WO_RESET == 1)
                is_after_reset = 1;

            r = fread(in_features, sizeof(float), NB_TOTAL_FEATURES, fin1);
            if (feof(fin1)) break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
//            remove_hiss(features);
            if (is_reset_frame)
            {
                lpcnet_copy(net2, net1);
                srand(net2->frame_count + 23);
                lpcnet_synthesize(net2, features, pcm2, LPCNET_FRAME_SIZE, 0);
            }
            srand(net1->frame_count + 23);
            lpcnet_synthesize(net1, features, pcm1, LPCNET_FRAME_SIZE, is_reset_frame);
            if (is_reset_frame)
            {
                float cur_l1_dist;
                float min_l1_dist = 0.0;
                for (j = 0; j < SIMILARITY_WINDOW; j++)
                    min_l1_dist += abs(pcm1[j] - pcm2[j]);
                for (i = 1; i <= MAX_SHIFT; i++)
                {
                    cur_l1_dist = 0.0;
                    for (j = i; j < i + SIMILARITY_WINDOW; j++)
                        cur_l1_dist += abs(pcm1[j] - pcm2[j-i]);
                    if (cur_l1_dist < min_l1_dist)
                    {
                        min_l1_dist = cur_l1_dist;
                        shift = i;
                    }
                }
                net1->last_shift = shift;
                memcpy(net1->last_smooth, pcm2, FRAME_SIZE*sizeof(short));
                memcpy(net1->last_hard, pcm1, FRAME_SIZE*sizeof(short));
            }
            if (is_after_reset)
            {
                shift = net1->last_shift;
                for (i = 0; i < FRAME_SIZE - shift; i++)
                {
                    float right = pow(i/((float)FRAME_SIZE), SMOOTH_FACTOR);
                    float left = 1.0 - right;
                    pcm2[i] = (left*net1->last_smooth[i] + right*net1->last_hard[i+shift]);
                }
                fwrite(pcm2, sizeof(short), FRAME_SIZE - shift, fout);
                for (i = 0; i < shift; i++)
                {
                    float right = pow((FRAME_SIZE-shift+i)/((float)FRAME_SIZE), SMOOTH_FACTOR);
                    float left = 1.0 - right;
                    pcm1[i] = (left*net1->last_smooth[FRAME_SIZE-shift+i] + right*pcm1[i]);
                }
                fwrite(pcm1, sizeof(short), FRAME_SIZE, fout);
            }
            else if (!is_reset_frame)
                fwrite(pcm1, sizeof(short), FRAME_SIZE, fout);
        }
//        insert_silent_frames(fout);
        printf("Number of reset frames is %d\n", reset_count);
        lpcnet_destroy(net1);
        lpcnet_destroy(net2);
    }
    else if (mode == MODE_RESET)
    {
        LPCNetState *net;
        net = lpcnet_create();
        srand(time(0));
        int frames_after_reset_count = 0;
        while (1)
        {
            float in_features[NB_TOTAL_FEATURES];
            short reset_frame[1];
            float u;
            r = fread(in_features, sizeof(in_features[0]), NB_TOTAL_FEATURES, fin1);
            if (feof(fin1)) break;

            reset_frame[0] = 0;
            if (frames_after_reset_count >= MIN_FRAMES_WO_RESET)
            {
                u = (float)rand()/RAND_MAX;
                if (u < RESET_PROBABILITY)
                {
                    reset_frame[0] = 1;
                    frames_after_reset_count = 0;
                }
                else
                    frames_after_reset_count++;
            }
            else
                frames_after_reset_count++;
            fwrite(reset_frame, sizeof(reset_frame[0]), 1, fout);
        }
        lpcnet_destroy(net);
    }
    else if (mode == MODE_SYNTHESIS_FAKE)
    {
        LPCNetState *net;
        net = lpcnet_create();
        printf("Synthesis...\n");
        while (1)
        {
            float in_features[NB_TOTAL_FEATURES];
            float features[NB_FEATURES];
            short pcm[LPCNET_FRAME_SIZE];
            short is_reset_frame[1];
            r = fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin1);
            r = fread(is_reset_frame, sizeof(is_reset_frame[0]), 1, fin2);
            if (feof(fin1) && feof(fin2)) break;
            if (feof(fin1) || feof(fin2)) 
            {
                printf("Error: %s and %s cover different number of frames.\n", argv[2], argv[3]);
                break;
            }
            if (is_reset_frame[0] != 0 && is_reset_frame[0] != 1)
            {
                printf("Incorrect input in %s\n", argv[3]);
                break;
            }
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
            lpcnet_synthesize(net, features, pcm, LPCNET_FRAME_SIZE, is_reset_frame[0]);
            fwrite(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, fout);
        }
        lpcnet_destroy(net);
    }
    else if (mode == MODE_ERROR)
    {
        int frame_num = 0;
        int check_next_frame = 0;
        float loss, loss_prev;
        float error[2];
        while (1)
        {
            float in_features_real[NB_TOTAL_FEATURES];
            float features_real[18];
            float in_features_fake[NB_TOTAL_FEATURES];
            float features_fake[18];
            short is_reset_frame[1];
            r = fread(in_features_real, sizeof(features_real[0]), NB_TOTAL_FEATURES, fin1);
            r = fread(in_features_fake, sizeof(features_fake[0]), NB_TOTAL_FEATURES, fin2);
            r = fread(is_reset_frame, sizeof(is_reset_frame[0]), 1, fin3);
            if (feof(fin1) && feof(fin2) && feof(fin3)) break;
            if (feof(fin1) || feof(fin2) || feof(fin3))
            {
                printf("Error: %s, %s and %s cover different number of frames.\n", argv[2], argv[3], argv[4]);
                break;
            }
            if (is_reset_frame[0] != 0 && is_reset_frame[0] != 1)
            {
                printf("Incorrect input in %s\n", argv[4]);
                break;
            }
            RNN_COPY(features_real, in_features_real, 18);
            RNN_COPY(features_fake, in_features_fake, 18);

            error[0] = 0.0;
            error[1] = 0.0;
            if (check_next_frame)
            {
                loss = get_loss(features_real, features_fake);
                error[0] = loss_prev + loss;
                error[1] = 1.0;
            }

            if (is_reset_frame[0])
            {
                loss_prev = get_loss(features_real, features_fake);
                check_next_frame = 1;
            }
            else
                check_next_frame = 0;
            if (frame_num > 0)
                fwrite(error, sizeof(error[0]), 2, fout);
            frame_num++;
        }
        error[0] = 0.0;
        error[1] = 0.0;
        fwrite(error, sizeof(error[0]), 2, fout);
    }
    else
        fprintf(stderr, "Unknown action.\n");
    fclose(fin1);
    if (mode == MODE_SYNTHESIS_FAKE || mode == MODE_ERROR)
        fclose(fin2);
    if (mode == MODE_ERROR)
        fclose(fin3);
    fclose(fout);
    return 0;
}
