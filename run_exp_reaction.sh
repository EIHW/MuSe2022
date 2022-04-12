#!/bin/sh

FEATURES="eGeMAPS"
#same for each feature type ['FAUs','DeepSpectrum','VGGFace2','eGeMAPS']
SEEDS=5
BATCHSIZE=64
EPOCHS=30
WINLEN=500
HOPLEN=250
RNNLAYERS=3
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

#run full exp (will not run without test labels)
#python3 main.py --task reaction --feature "$FEATURES" --batch_size "$BATCHSIZE" --normalize --epochs "$EPOCHS" --n_seeds "$SEEDS" --rnn_n_layers "$RNNLAYERS" --use_gpu --cache --win_len "$WINLEN" --hop_len "$HOPLEN" --result_csv results_csv/"$TIMESTAMP"_"$FEATURES"_"$SEEDS"_"$BATCHSIZE"_"$EPOCHS"_"$WINLEN"_"$HOPLEN"_"$RNNLAYERS".csv

#run on val and store predictions
python3 main.py --task reaction --feature "$FEATURES" --batch_size "$BATCHSIZE" --normalize --epochs "$EPOCHS" --n_seeds "$SEEDS" --rnn_n_layers "$RNNLAYERS" --use_gpu --cache --win_len "$WINLEN" --hop_len "$HOPLEN" --predict
