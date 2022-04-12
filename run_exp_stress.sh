# VALENCE

# EGEMAPS

python3 main.py --task stress --emo_dim valence --d_rnn 128 --rnn_n_layers 2 --lr 0.005 --feature egemaps --normalize --cache --result_csv  baseline_results/stress_v.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# DS

python3 main.py --task stress --emo_dim valence --d_rnn 64 --rnn_n_layers 2 --lr 0.002 --feature ds --cache --result_csv  baseline_results/stress_v.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# FAU

python3 main.py --task stress --emo_dim valence --d_rnn 32 --rnn_n_layers 4 --lr 0.002 --feature faus --cache --result_csv  baseline_results/stress_v.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# VGGFACE

python3 main.py --task stress --emo_dim valence --d_rnn 64 --rnn_n_layers 4 --lr 0.005 --feature vggface2 --cache --result_csv  baseline_results/stress_v.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# BERT
python3 main.py --task stress --emo_dim valence --d_rnn 64 --rnn_n_layers 4 --lr 0.002 --feature bert-4 --cache --result_csv  baseline_results/stress_v_newtext.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# biosignals
python3 main.py --task stress --emo_dim valence --d_rnn 16 --rnn_n_layers 4 --lr 0.005 --rnn_bi --feature biosignals --cache --result_csv  baseline_results/stress_signals.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5


# AROUSAL

# EGEMAPS

python3 main.py --task stress --emo_dim physio-arousal --d_rnn 128 --rnn_n_layers 4 --lr 0.001 --feature egemaps --normalize --cache --result_csv  baseline_results/stress_a.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# DS

python3 main.py --task stress --emo_dim physio-arousal --d_rnn 128 --rnn_n_layers 2 --rnn_bi --lr 0.0005 --feature ds --cache --result_csv  baseline_results/stress_a.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# FAU

python3 main.py --task stress --emo_dim physio-arousal --d_rnn 64 --rnn_n_layers 2 --lr 0.001 --feature faus --cache --result_csv  baseline_results/stress_a.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# VGGFACE

python3 main.py --task stress --emo_dim physio-arousal --d_rnn 64 --rnn_n_layers 2 --lr 0.001 --feature vggface2 --cache --result_csv  baseline_results/stress_a.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# BERT

python3 main.py --task stress --emo_dim physio-arousal --d_rnn 64 --rnn_n_layers 4 --lr 0.005 --feature bert-4 --cache --result_csv  baseline_results/stress_a_newtext.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5

# biosignals
python3 main.py --task stress --emo_dim physio-arousal --d_rnn 16 --rnn_n_layers 4 --lr 0.005 --rnn_bi --feature biosignals --cache --result_csv  baseline_results/stress_signals.csv --n_seeds 20 --seed 101 --batch_size 256  --early_stopping_patience 15 --reduce_lr_patience 5
