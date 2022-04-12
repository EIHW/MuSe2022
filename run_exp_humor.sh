# EGEMAPS

python3 main.py --task humor --d_rnn 64 --rnn_n_layers 1 --lr 0.001 --feature egemaps --normalize --cache --result_csv  baseline_results/humor.csv --n_seeds 5 --seed 101 --batch_size 256  --early_stopping_patience 3 --reduce_lr_patience 2

# DS

python3 main.py --task humor --d_rnn 64 --rnn_bi --rnn_n_layers 2 --lr 0.001 --feature ds --cache --result_csv  baseline_results/humor.csv --n_seeds 5 --seed 101 --batch_size 256  --early_stopping_patience 3 --reduce_lr_patience 2

# FAU

python3 main.py --task humor --d_rnn 256 --rnn_bi --rnn_n_layers 4 --lr 0.0001 --feature faus --cache --result_csv  baseline_results/humor.csv --n_seeds 5 --seed 101 --batch_size 256  --early_stopping_patience 3 --reduce_lr_patience 2

# VGGFACE

python3 main.py --task humor --d_rnn 64 --rnn_bi --rnn_n_layers 2 --lr 0.0001 --feature vggface2 --cache --result_csv  baseline_results/humor.csv --n_seeds 5 --seed 101 --batch_size 256  --early_stopping_patience 3 --reduce_lr_patience 2