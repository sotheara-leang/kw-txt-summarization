# Topic-Aware Abstractive Summarization

Integrate [Keywords attention mechanism](http://tcci.ccf.org.cn/conference/2018/papers/EV37.pdf) to [A Deep Reinforced Model for Abstraction Summarization](https://arxiv.org/pdf/1705.04304.pdf)

### Model Description
* Intra-temporal encoder and Intra-decoder attention for handling repeated words
* Pointer mechanism for handling out-of-vocabulary (OOV) words
* Self-critic policy gradient training along with MLE training
* Sharing decoder weights with word embedding

### Setup Project Env
* Create a virtual environment: `virtualenv -p python3 venv`
* Configure venv path in `start.sh`
* Start/Stop venv: `source start.sh`, `source stop.sh`
* Install libraries: `pip install -r requirement.txt`

### Server SSH Connetion
* Configure ssh user (eg. `user@server`) in `ssh.sh`
* Connect to server: `./ssh.sh connect 8888`
* Disconnect from server: `./ssh.sh disconnect 8888`

### Dataset
* Download CNN/Daily Mail Q/A dataset from [here](https://cs.nyu.edu/~kcho/DMQA/)
* Use these following utils to generate data:
    * `data/cnn_generate_data.py` to generate datasets (train, validation, test set)
    * `data/cnn_generate_vocab.py` to generate vocabulary from generated dataset
    * `data/cnn_process_data.py` to extract a set of examples from given dataset

### Glove
* Download Glove word embedding from [here](https://nlp.stanford.edu/projects/glove/)
* Use `data/glove_process_data.py` to generate the embedding file to be used for model

### Configuration
* All configurations for training and evaluating can be found in `main/conf` folder:
    * `main/conf/eval` for evaluation
    * `main/conf/train` for training
 
#### Common
| Parameter | Description |
|-----|-----|
|emb-size|Size of word embedding|
|emb-file|Glove word embedding. If not set, the embedding will be learned during training|
|enc-hidden-size|Size of encoder hidden state|
|dec-hidden-size|Size of decoder hidden state|
|max-enc-steps|Maximum length of article|
|max-dec-steps|Maximum length of summary|
|vocab-size|Size of vocabulary|
|vocab-file|Vocabulary file|
|intra-dec-attn|To enable intra-decoder attention|
|pointer-generator|To enable Pointer-Generator|
|share-dec-weight|To enable sharing decoder weights with word embedding|
|device|Device to be used (e.g. cpu, cuda:0)|
|logging||
|&nbsp;&nbsp;&nbsp;&nbsp;enable|To enable logging|
|&nbsp;&nbsp;&nbsp;&nbsp;conf-file|Path of logging config file. Default logging.yml at the same directory of config.yml|

#### Train
| Parameter | Description |
|-----|-----|
|epoch|Number of epoch|
|batch-size|Size of batch|
|log-batch|To enable logging each batch|
|log-batch-interval|Number of every batch to be logged|
|clip-gradient-max-norm|Maximum value of gradient|
|lr|Learning rate|
|lr-decay|Ratio to reduce learning rate|
|lr-decay-epoch|To update learning rate based on the `lr-decay`|
|ml||
|&nbsp;&nbsp;&nbsp;&nbsp;enable|To enable ML training|
|&nbsp;&nbsp;&nbsp;&nbsp;forcing-ratio|Ratio of teacher forcing|
|&nbsp;&nbsp;&nbsp;&nbsp;forcing-decay|Ratio to reduce `forcing-ratio`|
|rl||
|&nbsp;&nbsp;&nbsp;&nbsp;enable|To enable RL training|
|&nbsp;&nbsp;&nbsp;&nbsp;transit-epoch|To define which epoch to start RL training|
|&nbsp;&nbsp;&nbsp;&nbsp;transit-decay|Ratio to decrease the flag to enable RL training|
|&nbsp;&nbsp;&nbsp;&nbsp;weight|Weight of RL|
|eval|To evaluate the training set after finishing training|
|tb||
|&nbsp;&nbsp;&nbsp;&nbsp;enable|To enable TensorBoard logging|
|&nbsp;&nbsp;&nbsp;&nbsp;log-batch|To log every batch|
|&nbsp;&nbsp;&nbsp;&nbsp;log-dir|Directory to write logging file|
|article-file|Article file|
|keyword-file|Keyword file|
|summary-file|Summary file|
|load-model-file|Path to load pre-trained model|
|save-model-file|Path to save model (including file name)|
|save-model-per-epoch|Number of every epoch to save the model|