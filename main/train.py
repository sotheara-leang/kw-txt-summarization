from rouge import Rouge
import random
import torch.nn.functional as f
from torch.distributions import Categorical

from main.data.giga import *
from main.seq2seq import Seq2Seq
from main.common.batch import *
from main.common.util.file_util import FileUtil


class Train(object):

    def __init__(self):
        self.epoch = conf.get('train:epoch')
        self.rl_weight = conf.get('train:rl-weight')
        self.forcing_ratio = conf.get('train:forcing_ratio')
        self.forcing_decay = conf.get('train:forcing_decay')
        self.rl_transit_epoch = conf.get('train:rl_transit_epoch')
        self.rl_transit_decay = conf.get('train:rl_transit_decay')

        self.vocab = Vocab(FileUtil.get_file_path(conf.get('train:vocab-file')))

        self.seq2seq = cuda(Seq2Seq(self.vocab))

        self.batch_initializer = BatchInitializer(self.vocab, conf.get('max-enc-steps'))

        self.dataloader = GigaDataLoader(FileUtil.get_file_path(conf.get('train:article-file')), FileUtil.get_file_path(conf.get('train:summary-file')), conf.get('train:batch-size'))

        self.optimizer = t.optim.Adagrad(self.seq2seq.parameters(), lr=conf.get('train:learning_rate'))

    def train_batch(self, batch, epoch_counter):
        self.seq2seq.train()

        rouge       = Rouge()
        batch_size  = len(batch.articles_len)
        dec_input   = cuda(t.tensor([TK_START_DECODING.idx] * batch_size))  # B

        x = self.seq2seq.embedding(batch.articles)  # B, L, E

        enc_outputs, (enc_hidden_n, _) = self.seq2seq.encoder(x, batch.articles_len)  # (B, L, 2H), (B, 2H)

        # ML
        output = self.train_ml(enc_outputs, enc_hidden_n, dec_input, batch.extend_vocab, batch.summaries, epoch_counter)

        ml_loss = t.sum(output[1], dim=1) / t.sum(output[1] != 0, dim=1).float()
        ml_loss = t.mean(ml_loss)

        # transit reinforced learning

        if self.rl_transit_epoch >= 0:
            enable_rl = epoch_counter == self.rl_transit_epoch
        else:
            enable_rl = max(0, 1 - self.rl_transit_decay * epoch_counter) == 0

        if enable_rl:
            # sampling
            sample_output = self.train_rl(enc_outputs, enc_hidden_n, dec_input, batch.extend_vocab,  batch.summaries, True)

            # greedy search
            with t.autograd.no_grad():
                baseline_output = self.train_rl(enc_outputs, enc_hidden_n, dec_input, batch.extend_vocab,  batch.summaries, False)

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

            # loss

            rl_loss = t.sum(sample_output[1], dim=1) / t.sum(sample_output[1] != 0, dim=1).float()
            rl_loss = (baseline_scores - sample_scores) * rl_loss
            rl_loss = t.mean(rl_loss)

            # reward

            reward = t.mean(sample_scores - baseline_scores)

        else:
            rl_loss = cuda(t.zeros(1))
            reward = 0

        # total loss

        rl_weight = self.rl_weight if enable_rl else 0

        loss = rl_weight * rl_loss + (1 - rl_weight) * ml_loss

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss, ml_loss, rl_loss, reward, enable_rl

    def train_ml(self, enc_outputs, dec_hidden, dec_input, extend_vocab, target_y, epoch_counter):
        batch_size          = len(enc_outputs)
        y                   = None  # B, T
        loss                = None  # B, T
        enc_temporal_score  = None
        pre_dec_hiddens     = None  # B, T, 2H
        stop_decoding_mask  = cuda(t.zeros(batch_size))     # B
        max_ovv_len         = max([idx for vocab in extend_vocab for idx in vocab if idx == TK_UNKNOWN.idx] + [0] * len(extend_vocab))

        for i in range(target_y.size(1)):
            # decoding
            vocab_dist, dec_hidden, _, _, enc_temporal_score = self.seq2seq.decode(
                dec_input,
                dec_hidden,
                pre_dec_hiddens,
                enc_outputs,
                enc_temporal_score,
                extend_vocab,
                max_ovv_len)

            _, dec_output = t.max(vocab_dist, dim=1)

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            # loss

            step_loss = f.nll_loss(t.log(vocab_dist + 1e-12), target_y[:, i], reduction='none', ignore_index=TK_PADDING.idx)  # B

            loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)  # B, L

            # masking decoding

            stop_decoding_mask[(stop_decoding_mask == 0) + (dec_output == TK_STOP_DECODING.idx) == 2] = 1

            if len(stop_decoding_mask[stop_decoding_mask == 1]) == len(stop_decoding_mask):
                break

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

            # teacher forcing

            if self.forcing_ratio == 1:
                dec_input = target_y[:, i]
            else:
                forcing_ratio = max(0, self.forcing_ratio - self.forcing_decay * epoch_counter)

                use_ground_truth = t.randn(batch_size) < forcing_ratio  # B
                use_ground_truth = cuda(use_ground_truth.long())

                dec_input = use_ground_truth * target_y[:, i] + (1 - use_ground_truth) * dec_output  # B

            # if next input is oov, change it to UNKNOWN_TOKEN

            is_oov = (dec_input >= self.vocab.size()).long()

            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN.idx

        return y, loss

    def train_rl(self, enc_outputs, dec_hidden, dec_input, target_y, extend_vocab, sampling):
        batch_size          = len(enc_outputs)
        y                   = None  # B, T
        loss                = None  # B, T
        enc_temporal_score  = None
        pre_dec_hiddens     = None  # B, T, 2H
        stop_decoding_mask  = cuda(t.zeros(batch_size))
        dec_len             = self.max_dec_steps if target_y is None else target_y.size(1)
        max_ovv_len         = max([idx for vocab in extend_vocab for idx in vocab if idx == TK_UNKNOWN.idx] + [0] * len(extend_vocab))

        for i in range(dec_len):
            # decoding
            vocab_dist, dec_hidden, _, _, enc_temporal_score = self.seq2seq.decode(
                dec_input,
                dec_hidden,
                pre_dec_hiddens,
                enc_outputs,
                enc_temporal_score,
                extend_vocab,
                max_ovv_len)

            # sampling
            if sampling:
                sampling_dist = Categorical(vocab_dist)
                dec_output = sampling_dist.sample()

                step_loss = sampling_dist.log_prob(dec_output)
            else:
                # greedy search
                _, dec_output = t.max(vocab_dist, dim=1)

                step_loss = f.nll_loss(t.log(vocab_dist + 1e-12), target_y[:, i], reduction='none', ignore_index=TK_PADDING.idx)  # B

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)  # B, L

            stop_decoding_mask[(stop_decoding_mask == 0) + (dec_output == TK_STOP_DECODING.idx) == 2] = 1

            # stop when all masks are 1
            if len(stop_decoding_mask[stop_decoding_mask == 1]) == len(stop_decoding_mask):
                break

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

            dec_input = dec_output

        return y, loss

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
                loss, ml_loss, rl_loss, samples_award, enable_rl = self.train_batch(batch, i+1)

                total_loss += loss.item()
                total_ml_loss += ml_loss.item()
                total_rl_loss += rl_loss.item()
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
            if enable_rl:
                logger.debug('rl-loss\t=\t%.3f,\t reward=%.3f', rl_loss_avg, samples_reward_avg)
            else:
                logger.debug('rl-loss\t=\tNA')

        # save model
        model_path = FileUtil.get_file_path(conf.get('train:model-file'))

        logger.debug('save model into : ' + model_path)

        t.save(self.seq2seq.state_dict(), FileUtil.get_file_path(conf.get('train:model-file')))

    def evaluate(self):
        self.seq2seq.eval()

        article = 'south korea on monday announced sweeping tax reforms , including income and corporate tax cuts to boost growth by stimulating sluggish private consumption and business investment .'

        summary = self.seq2seq.summarize(article)

        print(summary)


if __name__ == "__main__":
    train = Train()
    train.run()

    train.evaluate()
