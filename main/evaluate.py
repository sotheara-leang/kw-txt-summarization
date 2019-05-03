from rouge import Rouge
from tensorboardX import SummaryWriter
import os
import time
import datetime
import argparse

from main.data.dataloader import *
from main.seq2seq import Seq2Seq
from main.common.batch import *
from main.common.util.file_util import FileUtil
from main.common.simple_vocab import SimpleVocab


class Evaluate(object):

    def __init__(self):
        self.logger                     = getLogger(self)

        self.max_enc_steps              = conf.get('max-enc-steps')
        self.max_dec_steps              = conf.get('max-dec-steps')
       
        self.batch_size                 = conf.get('eval:batch-size')
        self.log_batch                  = conf.get('eval:log-batch')
        self.log_batch_interval         = conf.get('eval:log-batch-interval', -1)

        self.tb_log_dir                 = conf.get('eval:tb-log-dir')

        self.pointer_generator          = conf.get('pointer-generator')


        self.vocab = SimpleVocab(FileUtil.get_file_path(conf.get('vocab-file')), conf.get('vocab-size'))

        self.seq2seq = cuda(Seq2Seq(self.vocab))

        self.batch_initializer = BatchInitializer(self.vocab, self.max_enc_steps, self.max_dec_steps, self.pointer_generator)

        self.data_loader = DataLoader(FileUtil.get_file_path(conf.get('eval:article-file')),
                                      FileUtil.get_file_path(conf.get('eval:summary-file')),
                                      FileUtil.get_file_path(conf.get('eval:keyword-file')), self.batch_size)

        if self.tb_log_dir is not None:
            self.tb_writer = SummaryWriter(FileUtil.get_file_path(self.tb_log_dir))

    def evaluate(self):
        self.logger.debug('>>> evaluation:')

        self.seq2seq.eval()

        rouge           = Rouge()
        total_scores    = []
        total_eval_time = time.time()
        batch_counter   = 0

        while True:
            eval_time = time.time()

            batch = self.data_loader.next_batch()

            if batch is None:
                break

            batch = self.batch_initializer.init(batch)

            max_ovv_len = max([len(oov) for oov in batch.oovs])

            # prediction

            output = self.seq2seq(batch.articles, batch.articles_len, batch.extend_vocab_articles, max_ovv_len, batch.keywords)

            gen_summaries = []
            for idx, summary in enumerate(output.tolist()):
                gen_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

            reference_summaries = batch.original_summaries

            # calculate rouge score

            scores = rouge.get_scores(list(gen_summaries), list(reference_summaries))
            scores = [score["rouge-l"]["f"] for score in scores]

            # logging batch

            avg_sores = sum(scores) / len(scores)

            eval_time = time.time() - eval_time

            if self.log_batch_interval <= 0 or (batch_counter + 1) % self.log_batch_interval == 0:
                self.logger.debug('BAT\t%d:\t\tavg rouge_l score=%.3f\t\ttime=%s', batch_counter + 1, avg_sores, str(datetime.timedelta(seconds=eval_time)))

            total_scores.append(avg_sores)

            batch_counter += 1

        avg_score = sum(total_scores) / len(total_scores)

        total_eval_time = time.time() - total_eval_time

        self.logger.debug('examples: %d', len(total_scores))
        self.logger.debug('avg rouge-l score: %.3f', avg_score)
        self.logger.debug('time\t:\t%s', str(datetime.timedelta(seconds=total_eval_time)))

    def load_model(self):
        model_file = conf.get('eval:load-model-file')
        if model_file is None:
            return
        model_file = FileUtil.get_file_path(model_file)

        if os.path.isfile(model_file):
            self.logger.debug('>>> load pre-trained model from: %s', model_file)

            checkpoint = t.load(model_file)

            self.seq2seq.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.warning('>>> cannot load pre-trained model - file not exist: %s', model_file)

    def run(self):
        # display configuration
        self.logger.debug('>>> configuration: \n' + conf.dump().strip())

        # load pre-trained model`
        self.load_model()

        # evaluate
        self.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str)

    args = parser.parse_args()

    config_file = args.conf_file
    if config_file is not None:
        conf.merge(config_file)

    evaluation = Evaluate()
    evaluation.run()
