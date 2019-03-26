from rouge import Rouge

from main.data.giga import *
from main.seq2seq import Seq2Seq
from main.common.batch import *


class Train(object):

    def __init__(self):
        self.epoch = conf.get('train:epoch')

        self.vocab = Vocab(conf.get('train:vocab-file'))

        self.seq2seq = Seq2Seq(self.vocab)

        self.batch_initializer = BatchInitializer(self.vocab, conf.get('max-enc-steps'))

        self.dataloader = GigaDataLoader(FileUtil.get_file_path(conf.get('train:article-file')), FileUtil.get_file_path(conf.get('train:summary-file')), conf.get('train:batch-size'))

        self.optimizer = t.optim.Adagrad(self.seq2seq.parameters(), lr=conf.get('train:lr'))

        self.rl_weight = conf.get('train:rl-weight')

    def train_batch(self, batch):
        rouge = Rouge()

        self.optimizer.zero_grad()

        # ML
        output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries,
            batch.extend_vocab, batch.max_ovv_len, calculate_loss=True, teacher_forcing=True, greedy_search=False)

        # RL
        sample_output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries, batch.extend_vocab, batch.max_ovv_len, calculate_loss=True, greedy_search=False)

        baseline_output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries, batch.extend_vocab, batch.max_ovv_len)

        # convert decoded output to string

        sample_summaries = batch_output_ids2words(sample_output[0].tolist(), self.vocab, batch.oovs)
        sample_summaries = [' '.join(summary) for summary in sample_summaries]

        baseline_summaries = batch_output_ids2words(baseline_output[0].tolist(), self.vocab, batch.oovs)
        baseline_summaries = [' '.join(summary) for summary in baseline_summaries]

        reference_summaries = batch.original_summaries

        # calculate rouge score

        sample_scores = rouge.get_scores(list(sample_summaries), list(reference_summaries))
        sample_scores = t.tensor([score["rouge-l"]["f"] for score in sample_scores])

        basline_scores = rouge.get_scores(list(baseline_summaries), list(reference_summaries))
        basline_scores = t.tensor([score["rouge-l"]["f"] for score in basline_scores])

        # ml loss
        ml_loss = t.sum(output[1]) / len(output[1])

        # rl loss
        rl_loss = -(sample_scores - basline_scores) * sample_output[1]
        rl_loss = t.sum(rl_loss) / len(rl_loss)

        loss = self.rl_weight * rl_loss + (1 - self.rl_weight) * ml_loss

        loss.backward()

        self.optimizer.step()

    def run(self):
        with t.autograd.set_detect_anomaly(True):

            self.seq2seq.train()

            for i in range(self.epoch):
                logger.debug('>>> Epoch %i/%i <<<', i+1, self.epoch)

                batch_counter = 1

                while True:
                    logger.debug('Batch %i', batch_counter)

                    batch = self.dataloader.next()

                    if batch is None:
                        break

                    batch = self.batch_initializer.init(batch)

                    self.train_batch(batch)

                    return

                    batch_counter += 1

                self.dataloader.reset()


if __name__ == "__main__":
    train = Train()
    train.run()
