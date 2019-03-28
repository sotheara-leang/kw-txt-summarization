from rouge import Rouge

from main.data.giga import *
from main.seq2seq import Seq2Seq
from main.common.batch import *
from main.common.util.file_util import FileUtil


class Train(object):

    def __init__(self):
        self.epoch = conf.get('train:epoch')

        self.vocab = Vocab(FileUtil.get_file_path(conf.get('train:vocab-file')))

        self.seq2seq = cuda(Seq2Seq(self.vocab))

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
            batch.extend_vocab, batch.max_ovv_len, calculate_loss=True, teacher_forcing=True, greedy_search=True)

        # RL
        sample_output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries, batch.extend_vocab, batch.max_ovv_len, calculate_loss=True, greedy_search=False)

        with t.autograd.no_grad():
            baseline_output = self.seq2seq(
                batch.articles,
                batch.articles_len,
                batch.summaries, batch.extend_vocab, batch.max_ovv_len)

        # convert decoded output to string

        sample_summaries = []
        sample_outputs = sample_output[0].tolist()
        for idx, summary in enumerate(sample_outputs):
            sample_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

        baseline_summaries = []
        baseline_outputs = baseline_output[0].tolist()
        for idx, summary in enumerate(baseline_outputs):
            baseline_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

        reference_summaries = batch.original_summaries

        # calculate rouge score

        sample_scores = rouge.get_scores(list(sample_summaries), list(reference_summaries))
        sample_scores = cuda(t.tensor([score["rouge-l"]["f"] for score in sample_scores]))

        baseline_scores = rouge.get_scores(list(baseline_summaries), list(reference_summaries))
        baseline_scores = cuda(t.tensor([score["rouge-l"]["f"] for score in baseline_scores]))

        # ml loss

        ml_loss = t.sum(output[1], dim=1) / batch.summaries_len.float()
        ml_loss = t.mean(ml_loss)

        # rl loss

        rl_loss = t.sum(sample_output[1], dim=1) / batch.summaries_len.float()
        rl_loss = (baseline_scores - sample_scores) * rl_loss
        rl_loss = t.mean(rl_loss)

        # total loss
        loss = self.rl_weight * rl_loss + (1 - self.rl_weight) * ml_loss

        # do backward
        loss.backward()

        # update model weight
        self.optimizer.step()

        reward = t.mean(sample_scores)

        return loss, ml_loss, rl_loss, reward

    def run(self):
        config_dump = conf.dump()
        logger.debug('configuration: \n' + config_dump.strip())

        for i in range(self.epoch):
            logger.debug('============ Epoch %i/%i ============', i + 1, self.epoch)

            batch_counter = 1

            total_loss = 0
            total_ml_loss = 0
            total_rl_loss = 0
            total_samples_award = 0

            while True:
                # get next batch
                batch = self.dataloader.next()

                if batch is None:
                    break

                # init batch
                batch = self.batch_initializer.init(batch)

                # feed batch to model
                loss, ml_loss, rl_loss, samples_award = self.train_batch(batch)

                total_loss += loss
                total_ml_loss += ml_loss
                total_rl_loss += rl_loss
                total_samples_award += samples_award

                batch_counter += 1
            #
            self.dataloader.reset()

            loss_avg = total_loss / batch_counter
            ml_loss_avg = total_ml_loss / batch_counter
            rl_loss_avg = total_rl_loss / batch_counter
            samples_reward_avg = total_samples_award / batch_counter

            logger.debug('loss\t\t=\t%.3f',  loss_avg)
            logger.debug('ml-loss\t=\t%.3f', ml_loss_avg)
            logger.debug('rl-loss\t=\t%.3f,\t reward=%.3f', rl_loss_avg, samples_reward_avg)

        # save model
        model_path = FileUtil.get_file_path(conf.get('train:model-file'))

        logger.debug('save model into : ' + model_path)

        t.save(self.seq2seq.state_dict(), FileUtil.get_file_path(conf.get('train:model-file')))


if __name__ == "__main__":
    train = Train()
    train.run()
