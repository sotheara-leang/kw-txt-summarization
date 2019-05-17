from rouge import Rouge
import os
import time
import datetime
import argparse

from main.data.cnn_dataloader import *
from main.seq2seq import Seq2Seq
from main.common.batch import *
from main.common.util.file_util import FileUtil
from main.common.simple_vocab import SimpleVocab


class Evaluate(object):

    def __init__(self):
        self.logger = logger(self)

        self.max_enc_steps      = conf('max-enc-steps')
        self.max_dec_steps      = conf('max-dec-steps')

        self.batch_size         = conf('eval:batch-size')
        self.log_batch          = conf('eval:log-batch')
        self.log_batch_interval = conf('eval:log-batch-interval', -1)

        self.pointer_generator  = conf('pointer-generator')


        self.vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        self.seq2seq = cuda(Seq2Seq(self.vocab))

        self.batch_initializer = BatchInitializer(self.vocab, self.max_enc_steps, self.max_dec_steps,
                                                  self.pointer_generator)

        self.data_loader = CNNDataLoader(FileUtil.get_file_path(conf('train:article-file')),
                                      FileUtil.get_file_path(conf('train:summary-file')),
                                      FileUtil.get_file_path(conf('train:keyword-file')), self.batch_size)

    def evaluate(self):
        self.logger.debug('>>> evaluation:')

        self.seq2seq.eval()

        rouge = Rouge()
        total_scores = []
        total_eval_time = time.time()
        batch_counter = 0
        example_counter = 0

        while True:
            eval_time = time.time()

            batch = self.data_loader.next_batch()

            if batch is None:
                break

            batch = self.batch_initializer.init(batch)

            max_ovv_len = max([len(oov) for oov in batch.oovs])

            # prediction

            output, _ = self.seq2seq(batch.articles, batch.articles_len, batch.extend_vocab_articles, max_ovv_len,
                                     batch.keywords)

            gen_summaries = []
            for idx, summary in enumerate(output.tolist()):
                summary = [w for w in summary if w != TK_STOP['id']]

                gen_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

            reference_summaries = batch.original_summaries

            # calculate rouge score

            avg_score = rouge.get_scores(list(gen_summaries), list(reference_summaries), avg=True)
            avg_score = avg_score["rouge-l"]["f"]

            # logging batch

            eval_time = time.time() - eval_time

            if self.log_batch_interval <= 0 or (batch_counter + 1) % self.log_batch_interval == 0:
                self.logger.debug('BAT\t%d:\t\tavg rouge_l score=%.3f\t\ttime=%s', batch_counter + 1, avg_score,
                                  str(datetime.timedelta(seconds=eval_time)))

            total_scores.append(avg_score)

            batch_counter += 1
            example_counter += batch.size

        avg_score = sum(total_scores) / len(total_scores)

        total_eval_time = time.time() - total_eval_time

        self.logger.debug('examples: %d', example_counter)
        self.logger.debug('avg rouge-l score: %f', avg_score)
        self.logger.debug('time\t:\t%s', str(datetime.timedelta(seconds=total_eval_time)))

    def load_model(self):
        model_file = conf('eval:load-model-file')
        if model_file is None:
            return
        model_file = FileUtil.get_file_path(model_file)

        if os.path.isfile(model_file):
            self.logger.debug('>>> load pre-trained model from: %s', model_file)

            checkpoint = t.load(model_file)

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            self.logger.debug('epoch: %s', str(epoch))
            self.logger.debug('loss: %s', str(loss))

            self.seq2seq.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise Exception('>>> cannot load model - file not exist: %s', model_file)

    def run(self):
        try:
            # display configuration
            self.logger.debug('>>> configuration: \n' + conf().dump().strip())

            # load pre-trained model`
            self.load_model()

            # evaluate
            self.evaluate()
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str)

    args = parser.parse_args()

    AppContext(args.conf_file)

    evaluation = Evaluate()
    evaluation.run()
