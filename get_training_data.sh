PCM_DIR=$1
OUT_DIR=$2

for pcm in $PCM_DIR/*.pcm; do
  src/dump_data -train "$pcm" "$OUT_DIR/features_train.f32" "$OUT_DIR/data_train.i8"
done
