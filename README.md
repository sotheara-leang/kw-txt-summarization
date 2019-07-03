# Topic-Aware Abstractive Summarization

Integrate [Keywords attention mechanism](http://tcci.ccf.org.cn/conference/2018/papers/EV37.pdf) to [A Deep Reinforced Model for Abstraction Summarization](https://arxiv.org/pdf/1705.04304.pdf)

## Model Description
* Intra-temporal encoder and Intra-decoder attention for handling repeated words
* Pointer mechanism for handling out-of-vocabulary (OOV) words
* Self-critic policy gradient training along with MLE training
* Sharing decoder weights with word embedding

## Setup Project Env
* Create a virtual environment: `virtualenv -p python3 venv`
* Configure venv path in `start.sh`
* Start/Stop venv: `source start.sh`, `source stop.sh`

## Server SSH Connetion
* Configure ssh user (eg. `user@server`) in `ssh.sh`
* Connect to server: `./ssh.sh connect 8888`
* Disconnect from server: `./ssh.sh disconnect 8888`

## Dataset
* Download CNN/Daily Mail Q/A dataset from [here](https://cs.nyu.edu/~kcho/DMQA/)
* Use these following utils to generate data:
    * `data/cnn_generate_data.py` to generate datasets (train, validation, test set)
    * `data/cnn_generate_vocab.py` to generate vocabulary from processed dataset
    * `data/cnn_process_data.py` to extract a set of examples from given dataset

## Glove
* Download Glove word embedding from [here](https://nlp.stanford.edu/projects/glove/)
* Use `data/glove_process_data.py` to generate the embedding file to be used for model

## Configuration
* All configurations for training and evaluating can be found in `main/conf` folder:
    * `main/conf/eval` for evaluation
    * `main/conf/train` for training
