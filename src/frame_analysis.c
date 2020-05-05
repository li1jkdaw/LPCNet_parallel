#include <math.h>
#include <stdio.h>
#include <time.h>
#include "common.h"
#include "arch.h"
#include "lpcnet.h"
#include "lpcnet_private.h"
#include "freq.h"

#define MAX_FRAMES 10000
#define MIN_RESET_FRAME_DISTANCE 20

#define SILENCE_ENERGY_THRESHOLD 500.0
#define HIGH_TO_LOW_ENERGY_THRESHOLD 100.0
#define NUM_HIGH_BANDS 4
#define NUM_HIGH_BANDS_FOR_CLICK 3
#define NUM_LOW_BANDS 5
#define NUM_LOW_BANDS_FOR_CLICK 3
#define NUM_CONSEQ_SILENT_FRAMES 4


// const char *feature_filename - file with Tacotron (with postnet) outputs
// int *reset_frames - array with numbers of frames for which synthesis can be started from scratch

// get_reset_frames - fills this array and returns the number of such frames

int get_reset_frames(const char *feature_filename, int *reset_frames)
{
    FILE *ffeats;
    ffeats = fopen(feature_filename, "rb");
    float *signal_energy;
    float *unvoiced_stat;
    int num_reset_frames = 0;
    int num_frames = 0;
    int i;
    int r;

    signal_energy = (float*)malloc(MAX_FRAMES*sizeof(float));
    unvoiced_stat = (float*)malloc(MAX_FRAMES*sizeof(float));

    /* processing file with Tacotron (with postnet) outputs */
    while (1)
    {
        float features[NB_TOTAL_FEATURES];
        float bfccs[NB_BANDS];
        float bark_spectrum[NB_BANDS];
        float total_energy = 0.0;
        float total_highbands_energy = 0.0;
        float total_lowbands_energy = 0.0;

        r = fread(features, sizeof(float), NB_TOTAL_FEATURES, ffeats);
        if (feof(ffeats))
            break;
        RNN_CLEAR(&features[NB_BANDS], NB_BANDS);

        /* conversion from BFCC to BFSC */
        bfccs[0] = features[0] + 4;
        for (i = 1; i<NB_BANDS; i++)
            bfccs[i] = features[i];
        idct(bark_spectrum, bfccs);

        /* calculating energy of the signal*/        
        for (i = 0; i<NUM_HIGH_BANDS; i++)
            total_highbands_energy += pow(10.0, bark_spectrum[NB_BANDS-i-1]);
        for (i = 0; i<NUM_LOW_BANDS; i++)
            total_lowbands_energy += pow(10.0, bark_spectrum[i]);
        total_energy = total_highbands_energy + total_lowbands_energy;
        for (i = NUM_LOW_BANDS; i < NB_BANDS-NUM_HIGH_BANDS; i++)
            total_energy += pow(10.0, bark_spectrum[i]);

        signal_energy[num_frames] = total_energy;
        unvoiced_stat[num_frames] = total_highbands_energy/total_lowbands_energy;

        num_frames++;
    }
    fclose(ffeats);

    /* defining which frames contain silence or correspond to unvoiced sounds */
    int silence_count = 0;
    int last_reset_frame = -1000;
    int last_unvoiced_frame = -1000;
    int last_unvoiced_stat = HIGH_TO_LOW_ENERGY_THRESHOLD;
    i = MIN_RESET_FRAME_DISTANCE - NUM_CONSEQ_SILENT_FRAMES;
    while (i < num_frames - MIN_RESET_FRAME_DISTANCE)
    {
        i++;
        /* check whether i-th frame contains silence */
        if (signal_energy[i] < SILENCE_ENERGY_THRESHOLD)
            silence_count++;
        else
            silence_count = 0;

        /* check whether i-th frame corresponds to unvoiced sound */
        if (unvoiced_stat[i] > HIGH_TO_LOW_ENERGY_THRESHOLD)
            last_unvoiced_frame = i;
        else
            last_unvoiced_stat = HIGH_TO_LOW_ENERGY_THRESHOLD;

        /* reset only at steps with distance > MIN_RESET_FRAME_DISTANCE */
        if (i - last_reset_frame > MIN_RESET_FRAME_DISTANCE)
        {
            if (silence_count == NUM_CONSEQ_SILENT_FRAMES)
            {
                reset_frames[num_reset_frames] = i;
                last_reset_frame = i;
                num_reset_frames++;
                continue;
            }
            if (last_unvoiced_frame == i)
            {
                if (unvoiced_stat[i] < last_unvoiced_stat)
                {
                    last_unvoiced_stat = HIGH_TO_LOW_ENERGY_THRESHOLD;
                    reset_frames[num_reset_frames] = i;
                    last_reset_frame = i;
                    num_reset_frames++;
                    continue;
                }
                last_unvoiced_stat = unvoiced_stat[i];
            }
        }
    }

    free(signal_energy);
    free(unvoiced_stat);

    return num_reset_frames;
}


float get_loss(const float *features_real, const float *features_fake)
{
    float bfccs[NB_BANDS];
    float bark_spectrum[NB_BANDS];
    float log_energy_real[NB_BANDS];
    float log_energy_fake[NB_BANDS];
    float diff_low, diff_high;
    int i;

    /* conversion from BFCC to BFSC for real features */
    bfccs[0] = features_real[0] + 4;
    for (i = 1; i < NB_BANDS; i++)
        bfccs[i] = features_real[i];
    idct(bark_spectrum, bfccs);

    /* calculating energy of the real signal*/        
    for (i = 0; i < NB_BANDS; i++)
        log_energy_real[i] = bark_spectrum[i];

    /* conversion from BFCC to BFSC for fake features */
    bfccs[0] = features_fake[0] + 4;
    for (i = 1; i < NB_BANDS; i++)
        bfccs[i] = features_fake[i];
    idct(bark_spectrum, bfccs);

    /* calculating energy of the fake signal*/        
    for (i = 0; i < NB_BANDS; i++)
        log_energy_fake[i] = bark_spectrum[i];

    /* calculating how strong click is */
    diff_low = 0.0;
    for (i = 0; i < NUM_LOW_BANDS_FOR_CLICK; i++)
    {
        if (log_energy_fake[i] > log_energy_real[i] && log_energy_real[i] > 1.01)
        {
            float stat = (pow(log_energy_fake[i], 2.0)/log_energy_real[i]);
            if (diff_low < stat)
                diff_low = stat;
        }
    }
    
    diff_high = 0.0;
    for (i = NB_BANDS-NUM_HIGH_BANDS_FOR_CLICK; i < NB_BANDS; i++)
    {
        if (log_energy_fake[i] > log_energy_real[i] && log_energy_real[i] > 1.01)
        {
            float stat = log_energy_fake[i]/log_energy_real[i];
            if (diff_high < stat)
                diff_high = stat;
        }
    }

    return pow(diff_low + diff_high, 2.0);
}
