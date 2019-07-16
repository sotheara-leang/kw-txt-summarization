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
* Download and extract CNN/Daily Mail Q/A dataset from [here](https://cs.nyu.edu/~kcho/DMQA/)
* Use these following utils to generate data:
    * `data/cnn_generate_data.py` to generate datasets (train, validation, test set)
        * Preprocess data - `python cnn_generate_data.py --opt preprocess --input_dir`
            * `input_dir` - CNN/Daily Mail dataset folder
        * Generate data: `python cnn_generate_data.py --opt generate --input_dir --output_dir [--validation_test_fraction]`
            * `input_dir` - CNN/Daily Mail dataset folder
            * `output_dir` - output folder to write the generated files
            * `validation_test_fraction` - fraction of validation and test set. Default: 0.10
    * `data/cnn_generate_vocab.py` to generate vocabulary from generated dataset
        * `python cnn_generate_vocab.py --files article.txt summary.txt [--fname] [--max_vocab] [--dir_out]`
            * `fname` - vocabulary file name. Default: vocab.txt
            * `max_vocab` - maximum vocabulary words. Default: -1
            * `dir_out` - output directory. Default: data/extract
    * `data/cnn_process_data.py` to extract a set of examples from given dataset

### Word Embedding
* Download Glove word embedding from [here](https://nlp.stanford.edu/projects/glove/)
* Use `data/glove_process_data.py` to generate the embedding file to be used for model
    * `python glove_process_data.py --file glove.6B.100d.txt [--dir_out] [--fname]`
        * `dir_out` - output directory. Default: data/extract 
        * `fname` - output file name. Default: embedding.bin

### Configuration
* All configurations for training and evaluating can be found in `main/conf` folder. File `conf.yml` is for parameter configuration and `logging.yml` is for logging configuration.
    * `main/conf/eval` for evaluation
    * `main/conf/train` for training
 
#### Common Parameters
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
|share-dec-weight|To enable sharing decoder weights|
|device|Device to be used (e.g. cpu, cuda:0)|
|logging||
|&nbsp;&nbsp;&nbsp;&nbsp;enable|To enable logging|
|&nbsp;&nbsp;&nbsp;&nbsp;conf-file|Path of logging config file. Default: logging.yml in the same directory of config.yml|

#### Training Parameters
| Parameter | Description |
|-----|-----|
|epoch|Number of epoch|
|batch-size|Size of batch|
|log-batch|To enable logging each batch|
|log-batch-interval|Number of every batchs to be logged|
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
|tune-emb|To tune pretrained word embedding|
|tb||
|&nbsp;&nbsp;&nbsp;&nbsp;enable|To enable TensorBoard logging|
|&nbsp;&nbsp;&nbsp;&nbsp;log-batch|To log each batch|
|&nbsp;&nbsp;&nbsp;&nbsp;log-dir|Directory to write logging file|
|article-file|Article file|
|keyword-file|Keyword file|
|summary-file|Summary file|
|load-model-file|Path to load pre-trained model|
|save-model-file|Path to save model (including file name)|
|save-model-per-epoch|Number of every epoch to save the model|

#### Evaluation Parameters
| Parameter | Description |
|-----|-----|
|batch-size|Size of batch|
|log-batch|To enable logging each batch|
|log-batch-interval|Number of every batchs to be logged|
|article-file|Article file|
|keyword-file|Keyword file|
|summary-file|Summary file|
|load-model-file|Path to load pre-trained model|


### Running

From the root directory of project:

* Training: `python -m main.train [--conf_file]`
    * `conf_file` - training config file. Default: `main/conf/train/config.yml` 

* Evaluation: `python -m main.evaluate [--conf_file]`
    * `conf_file` - evaluation config file. Default: `main/conf/eval/config.yml`