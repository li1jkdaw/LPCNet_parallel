PCM_DIR=$1
OUT_DIR=$2

starttime=$(date +%s)
echo "Creating training dataset from audio records in $PCM_DIR"
for pcm_true in $PCM_DIR/*.pcm; do
  echo "Processing $pcm_true"
  ./dump_data -test "$pcm_true" "$OUT_DIR/features_true.f32"
  echo "Synthesizing in a standard way"
  ../build/LPCNet -synthesis_std "$OUT_DIR/features_true.f32" "$OUT_DIR/audio_wo_reset.pcm"
  ./dump_data -test "$OUT_DIR/audio_wo_reset.pcm" "$OUT_DIR/features_wo_reset.f32"
  ./dump_data -test "$OUT_DIR/audio_wo_reset.pcm" "$OUT_DIR/features.f32"
  echo "Defining reset frames from which sampling is started from scratch"
  ../build/LPCNet -reset "$OUT_DIR/features_wo_reset.f32" "$OUT_DIR/reset.msk"
  echo "Synthesizing with resets"
  ../build/LPCNet -synthesis_fake "$OUT_DIR/features_true.f32" "$OUT_DIR/reset.msk" "$OUT_DIR/audio_with_reset.pcm"
  ./dump_data -test "$OUT_DIR/audio_with_reset.pcm" "$OUT_DIR/features_with_reset.f32"
  echo "Calculating error"
  ../build/LPCNet -error "$OUT_DIR/features_wo_reset.f32" "$OUT_DIR/features_with_reset.f32" "$OUT_DIR/reset.msk" "$OUT_DIR/error.loss"
  echo "Deleting temporary files"
  rm "$OUT_DIR/reset.msk" "$OUT_DIR/audio_with_reset.pcm" "$OUT_DIR/audio_wo_reset.pcm" "$OUT_DIR/features_true.f32" "$OUT_DIR/features_wo_reset.f32" "$OUT_DIR/features_with_reset.f32"
  echo
done
endtime=$(date +%s)
echo "Elapsed time is $(($endtime - $starttime)) seconds"
