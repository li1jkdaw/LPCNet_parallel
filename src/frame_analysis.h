#ifndef FRAME_ANALYSIS_H
#define FRAME_ANALYSIS_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "stdio.h"
#include "lpcnet_private.h"

int get_reset_frames(const char *feature_filename, int *reset_frames);
float get_loss(const float *features_real, const float *features_fake);

#endif