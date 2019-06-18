import argparse
import datetime
import os
import time
import numpy as np
from rouge import Rouge

from main.common.batch import *
from main.common.simple_vocab import SimpleVocab
from main.common.util.file_util import FileUtil
from main.data.cnn_dataloader import *
from main.seq2seq import Seq2Seq
from main.common.glove.embedding import GloveEmbedding


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

        embedding = GloveEmbedding(FileUtil.get_file_path(conf('emb-file')), self.vocab) if conf('emb-file') is not None else None

        self.seq2seq = cuda(Seq2Seq(self.vocab, embedding))

        self.batch_initializer = BatchInitializer(self.vocab, self.max_enc_steps, self.max_dec_steps, self.pointer_generator)

        self.data_loader = CNNDataLoader(FileUtil.get_file_path(conf('eval:article-file')),
                                         FileUtil.get_file_path(conf('eval:summary-file')),
                                         FileUtil.get_file_path(conf('eval:keyword-file')), self.batch_size)

    def evaluate(self):
        self.logger.debug('>>> evaluation:')

        self.seq2seq.eval()

        rouge = Rouge()

        total_scores_1 = np.asarray([])
        total_scores_2 = np.asarray([])
        total_scores_l = np.asarray([])

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

            output, _ = self.seq2seq(batch.articles, batch.articles_len, batch.extend_vocab_articles, max_ovv_len, batch.keywords)

            t.cuda.empty_cache()

            gen_summaries = []
            for idx, summary in enumerate(output.tolist()):
                summary = [w for w in summary if w != TK_STOP['id']]

                gen_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

            reference_summaries = batch.original_summaries

            # calculate rouge score

            scores = rouge.get_scores(list(gen_summaries), list(reference_summaries))

            scores_1 = []
            scores_2 = []
            scores_l = []
            for score in scores:
                scores_1.append(score["rouge-1"]["f"])
                scores_2.append(score["rouge-2"]["f"])
                scores_l.append(score["rouge-l"]["f"])

            total_scores_1 = np.append(total_scores_1, scores_1, axis=0)
            total_scores_2 = np.append(total_scores_2, scores_2, axis=0)
            total_scores_l = np.append(total_scores_l, scores_l, axis=0)

            # logging batch

            eval_time = time.time() - eval_time

            if self.log_batch_interval <= 0 or (batch_counter + 1) % self.log_batch_interval == 0:
                avg_score_1 = np.mean(np.asarray(scores_1), axis=0)
                avg_score_2 = np.mean(np.asarray(scores_2), axis=0)
                avg_score_l = np.mean(np.asarray(scores_l), axis=0)

                self.logger.debug('BAT\t%d:\t\trouge-1=%.3f\t\trouge-2=%.3f\t\trouge-l=%.3f\t\ttime=%s',
                                  batch_counter + 1,
                                  avg_score_1, avg_score_2, avg_score_l,
                                  str(datetime.timedelta(seconds=eval_time)))

            batch_counter += 1
            example_counter += batch.size

        total_avg_score_1 = np.mean(total_scores_1, axis=0)
        total_std_score_1 = total_scores_1.std(axis=0)

        total_avg_score_2 = np.mean(total_scores_2, axis=0)
        total_std_score_2 = total_scores_2.std(axis=0)

        total_avg_score_l = np.mean(total_scores_l, axis=0)
        total_std_score_l = total_scores_l.std(axis=0)

        total_eval_time = time.time() - total_eval_time

        self.logger.debug('examples: %d', example_counter)
        self.logger.debug('avg rouge-1: %f\t, std rouge-1: %f', total_avg_score_1, total_std_score_1)
        self.logger.debug('avg rouge-2: %f\t, std rouge-2: %f', total_avg_score_2, total_std_score_2)
        self.logger.debug('avg rouge-l: %f\t, std rouge-l: %f', total_avg_score_l, total_std_score_l)

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
