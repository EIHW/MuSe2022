# MuSe-2022 Baseline Model: LSTM Regressor

[Homepage](https://www.muse-challenge.org) || [Baseline Paper](http://dx.doi.org/10.13140/RG.2.2.33203.91681)


## Sub-challenges and Results 
For details, please see the [Baseline Paper](http://dx.doi.org/10.13140/RG.2.2.33203.91681). If you want to sign up for the challenge, please fill the form 
[here](https://www.muse-challenge.org/challenge/participation) for MuSe-Humor and MuSe-Stress. For MuSe-Reaction, please contact competitions \[at\] hume.ai 

* MuSe-Humor: predicting presence/absence of humor in football press conference recordings. 
*Official baseline*: **.8480** AUC.

* MuSe-Reaction: predicting the intensity of seven emotions (Adoration, Amusement, Anxiety, Disgust, Empathic Pain, Fear,
Surprise). *Official baseline* : **.2801** mean Pearson's correlation over all seven classes.
* MuSe-Stress: regression on valence and arousal signals for persons in a stressed disposition. *Official baselines*:
**.4931** CCC for valence, **.4761** CCC for (physiological) arousal, **.4585** CCC as the mean of best CCC for arousal and 
best CCC for valence (*Combined*). Note that the *Combined* score will be used to determine the challenge winner.

## Installation
It is highly recommended to run everything in a Python virtual environment. Please make sure to install the packages listed 
in ``requirements.txt`` and adjust the paths in `config.py` (especially ``BASE_PATH``). 

You can then e.g. run the unimodal baseline reproduction calls in the ``*.sh`` file provided for each sub-challenge.

## Settings
The ``main.py`` script is used for training and evaluating models. Most important options:
* ``--task``: choose either `humor`, `reaction` or `stress` 
* ``--feature``: choose a feature set provided in the data (in the ``PATH_TO_FEATURES`` defined in ``config.py``). Adding 
``--normalize`` ensures normalization of features (recommended for eGeMAPS features).
* Options defining the model architecture: ``d_rnn``, ``rnn_n_layers``, ``rnn_bi``, ``d_fc_out``
* Options for the training process: ``--epochs``, ``--lr``, ``--seed``,  ``--n_seeds``, ``--early_stopping_patience``,
``--reduce_lr_patience``,   ``--rnn_dropout``, ``--linear_dropout``
* In order to use a GPU, please add the flag ``--use_gpu``
* Specific parameters for MuSe-Stress: ``emo_dim`` (``valence`` or ``physio-arousal``), ``win_len`` and ``hop_len`` for segmentation.

For more details, please see the ``parse_args()`` method in ``main.py``. 

## Reproducing the baselines 

### Unimodal results
For every challenge, a ``*.sh`` file is provided with the respective call (and, thus, configuration) for each of the precomputed features.
Moreover, you can directly load one of the provided checkpoints corresponding to the results in the baseline paper.
For MuSe-Humor, you can download the checkpoints [here](https://drive.google.com/drive/folders/14rBQ9ZKfClXK8z8JKTdxKGnLuxEdJS4Z?usp=sharing). 
The checkpoints for MuSe-Stress can be found [here](https://drive.google.com/drive/folders/1DYGEdH3WNNmu-ULTaO3RXnh_ALLA9QEv?usp=sharing).
Regarding MuSe-Reaction, the checkpoints are only available to registered participants. 
A checkpoint model can be loaded and evaluated as follows:

`` main.py --task humor --feature vggface2 --eval_model /your/checkpoint/directory/vggface2/model_102.pth`` 

Note that egemaps features must be normalized (``--normalize``).

### Fusion results 

#### Late Fusion (MuSe-Humor, MuSe-Stress)
The idea of the late fusion implementation is to treat the predictions to be fused as a new feature set that is 
stored in the feature directory (alongside with e.g. ``egemaps``, ``ds`` etc.). The script ``late_fusion.py`` creates 
such a feature set, given trained models. Example call:

`` python3 late_fusion_preparation.py --task humor --model_ids 2022-03-26-11-04_[ds]_[64_2_True_64]_[0.001_256] 2022-03-26-11-11_[vggface2]_[64_2_True_64]_[0.0001_256] --checkpoint_seeds 105 102``

The model ids are directories under ``MODEL_FOLDER/task`` (``MODEL_FOLDER`` is specified in ``config.py``).
For further details see the ``ArgumentParser`` in ``late_fusion_preparation.py``. 

This script will create a new feature set in the feature directory specified in ``config.py``. This feature set 
can then be used with the ``main.py`` script as any other feature.

#### Early Fusion (MuSe-Reaction)
Similarly, for early fusion, an extra feature set is created. The corresponding script is ``early_fusion_preparation.py``.
Example call: 
`` python3 early_fusion_preparation.py --task reaction --feature_sets VGGFace2 DeepSpectrum``

##  Citation:

The MuSe2022 baseline paper is available as a preprint [here](https://www.researchgate.net/publication/359875358_The_MuSe_2022_Multimodal_Sentiment_Analysis_Challenge_Humor_Emotional_Reactions_and_Stress)

```bibtex
@inproceedings{Christ22-TM2,
  title={The MuSe 2022 Multimodal Sentiment Analysis Challenge: Humor, Emotional Reactions, and Stress},
  author={Christ, Lukas and Amiriparian, Shahin and Baird, Alice and Tzirakis, Panagiotis and Kathan, Alexander and Müller, Niklas and Stappen, Lukas and Meßner, Eva-Maria and König, Andreas and Cowen, Alan and Cambria, Erik and Schuller, Bj\"orn W. },
  booktitle={Proceedings of the 3rd Multimodal Sentiment Analysis Challenge},
  year={2022},
  address = {Lisbon, Portugal},
  publisher = {Association for Computing Machinery},
  note = {co-located with ACM Multimedia 2022, to appear}
}

```

MuSe 2021 baseline paper:

```bibtex
@incollection{stappen2021muse,
  title={The MuSe 2021 multimodal sentiment analysis challenge: sentiment, emotion, physiological-emotion, and stress},
  author={Stappen, Lukas and Baird, Alice and Christ, Lukas and Schumann, Lea and Sertolli, Benjamin and Messner, Eva-Maria and Cambria, Erik and Zhao, Guoying and Schuller, Bj{\"o}rn W},
  booktitle={Proceedings of the 2nd on Multimodal Sentiment Analysis Challenge},
  pages={5--14},
  year={2021}
}

```
