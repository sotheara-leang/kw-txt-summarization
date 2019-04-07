from rouge import Rouge
import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical

from main.data.giga import *
from main.seq2seq import Seq2Seq
from main.common.batch import *
from main.common.util.file_util import FileUtil
from main.common.glove.vocab import GloveVocab
from main.common.simple_vocab import SimpleVocab
from main.common.glove.embedding import GloveEmbedding


class Train(object):

    def __init__(self):
        self.epoch                      = conf.get('train:epoch')
        self.batch_size                 = conf.get('train:batch-size')
        self.clip_gradient_max_norm     = conf.get('train:clip-gradient-max-norm')
        self.log_batch                  = conf.get('train:log-batch')
        self.lr                         = conf.get('train:lr')
        self.lr_decay_epoch             = conf.get('train:lr-decay-epoch')
        self.lr_decay                   = conf.get('train:lr-decay')

        self.ml_enable                  = conf.get('train:ml:enable')
        self.ml_forcing_ratio           = conf.get('train:ml:forcing-ratio')
        self.ml_forcing_decay           = conf.get('train:ml:forcing-decay')

        self.rl_enable                  = conf.get('train:rl:enable')
        self.rl_weight                  = conf.get('train:rl:weight')
        self.rl_transit_epoch           = conf.get('train:rl:transit-epoch')
        self.rl_transit_decay           = conf.get('train:rl:transit-decay')

        self.vocab = SimpleVocab(FileUtil.get_file_path(conf.get('train:vocab-file')), conf.get('vocab-size'))
        #self.vocab = GloveVocab(FileUtil.get_file_path(conf.get('train:vocab-file')))

        self.seq2seq = cuda(Seq2Seq(self.vocab))

        #self.seq2seq = cuda(Seq2Seq(self.vocab, GloveEmbedding(FileUtil.get_file_path(conf.get('train:emb-file')))))

        self.batch_initializer = BatchInitializer(self.vocab, conf.get('max-enc-steps'))

        self.dataLoader = GigaDataLoader(FileUtil.get_file_path(conf.get('train:article-file')),
                                         FileUtil.get_file_path(conf.get('train:summary-file')), self.batch_size)

        self.optimizer = t.optim.Adam(self.seq2seq.parameters(), lr=self.lr)

        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=TK_PADDING['id'])

    def train_batch(self, batch, epoch_counter):
        self.optimizer.zero_grad()

        rouge       = Rouge()
        dec_input   = cuda(t.tensor([TK_START_DECODING['id']] * self.batch_size))  # B

        ## encoding input

        x = self.seq2seq.embedding(batch.articles)  # B, L, E

        enc_outputs, (enc_hidden, enc_cell) = self.seq2seq.encoder(x, batch.articles_len)  # (B, L, 2H), (B, 2H)

        ## ML

        if self.ml_enable:
            output = self.train_ml(enc_outputs, enc_hidden, enc_cell, dec_input, batch.extend_vocab, batch.summaries, epoch_counter)

            ml_loss = t.sum(output[1], dim=1) / t.sum(output[1] != 0, dim=1).float()
            ml_loss = t.mean(ml_loss)
        else:
            ml_loss = cuda(t.zeros(1))

        ## RL

        rl_enable = self.rl_enable
        if rl_enable:
            if self.rl_transit_epoch >= 0:
                rl_enable = epoch_counter == self.rl_transit_epoch
            else:
                rl_enable = max(0, 1 - self.rl_transit_decay * epoch_counter) == 0

        if rl_enable:
            # sampling
            sample_output = self.train_rl(enc_outputs, enc_hidden, enc_cell, dec_input, batch.extend_vocab,  batch.summaries, True)

            # greedy search
            with t.autograd.no_grad():
                baseline_output = self.train_rl(enc_outputs, enc_hidden, enc_cell, dec_input, batch.extend_vocab,  batch.summaries, False)

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

        ## total loss

        rl_weight = self.rl_weight if rl_enable else 0

        loss = rl_weight * rl_loss + (1 - rl_weight) * ml_loss

        loss.backward()

        nn.utils.clip_grad_norm_(self.seq2seq.parameters(), self.clip_gradient_max_norm)

        self.optimizer.step()

        return loss, ml_loss, rl_loss, reward, rl_enable

    def train_ml(self, enc_outputs, dec_hidden, dec_cell, dec_input, extend_vocab, target_y, epoch_counter):
        y                   = None  # B, T
        loss                = None  # B, T
        enc_temporal_score  = None
        pre_dec_hiddens     = None  # B, T, 2H
        stop_decoding_mask  = cuda(t.zeros(self.batch_size))     # B
        max_ovv_len         = max([len(vocab) for vocab in extend_vocab])

        for i in range(target_y.size(1)):
            ## decoding
            vocab_dist, dec_hidden, dec_cell, _, _, enc_temporal_score = self.seq2seq.decode(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                enc_temporal_score,
                extend_vocab,
                max_ovv_len)

            ## loss

            step_loss = self.criterion(vocab_dist, target_y[:, i])  # B

            loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)  # B, L

            ## output

            _, dec_output = t.max(vocab_dist, dim=1)

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            ## masking decoding

            stop_decoding_mask[(stop_decoding_mask == 0) + (dec_output == TK_STOP_DECODING['id']) == 2] = 1

            if len(stop_decoding_mask[stop_decoding_mask == 1]) == len(stop_decoding_mask):
                break

            ## teacher forcing

            forcing_ratio = max(0, self.ml_forcing_ratio - self.ml_forcing_decay * epoch_counter)

            use_ground_truth = t.randn(self.batch_size) < forcing_ratio  # B
            use_ground_truth = cuda(use_ground_truth.long())

            dec_input = use_ground_truth * target_y[:, i] + (1 - use_ground_truth) * dec_output  # B

            ## if next decoder input is oov, change it to UNKNOWN_TOKEN

            is_oov = (dec_input >= self.vocab.size()).long()

            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN['id']

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

        return y, loss

    def train_rl(self, enc_outputs, dec_hidden, dec_cell, dec_input, target_y, extend_vocab, sampling):
        y                   = None  # B, T
        loss                = None  # B, T
        enc_temporal_score  = None
        pre_dec_hiddens     = None  # B, T, 2H
        stop_decoding_mask  = cuda(t.zeros(self.batch_size))
        dec_len             = self.max_dec_steps if target_y is None else target_y.size(1)
        max_ovv_len         = max([len(vocab) for vocab in extend_vocab])

        for i in range(dec_len):
            ## decoding
            vocab_dist, dec_hidden, dec_cell, _, _, enc_temporal_score = self.seq2seq.decode(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                enc_temporal_score,
                extend_vocab,
                max_ovv_len)

            ## sampling
            if sampling:
                sampling_dist = Categorical(vocab_dist, dim=1)
                dec_output = sampling_dist.sample()

                step_loss = sampling_dist.log_prob(dec_output)
            else:
                ## greedy search
                step_loss = self.criterion(vocab_dist, target_y[:, i])  # B

                _, dec_output = t.max(vocab_dist, dim=1)

            ## y & loss

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)  # B, L

            ## masking decoding

            stop_decoding_mask[(stop_decoding_mask == 0) + (dec_output == TK_STOP_DECODING['id']) == 2] = 1

            if len(stop_decoding_mask[stop_decoding_mask == 1]) == len(stop_decoding_mask):
                break

            dec_input = dec_output

            ## if next decoder input is oov, change it to UNKNOWN_TOKEN

            is_oov = (dec_input >= self.vocab.size()).long()

            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN['id']

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

        return y, loss

    def run(self):
        self.seq2seq.train()

        logger.debug('configuration: \n' + conf.dump().strip())

        criterion_scheduler = self.get_lr_scheduler(self.optimizer)

        for i in range(self.epoch):
            logger.debug('================= Epoch %i/%i =================', i + 1, self.epoch)

            batch_counter       = 0

            total_loss          = 0
            total_ml_loss       = 0
            total_rl_loss       = 0
            total_samples_award = 0

            criterion_scheduler.step()

            while True:

                # get next batch
                batch = self.dataLoader.next_batch()

                if batch is None:
                    break

                # init batch
                batch = self.batch_initializer.init(batch)

                # feed batch to model
                loss, ml_loss, rl_loss, samples_reward, enable_rl = self.train_batch(batch, i+1)

                if self.log_batch:
                    if enable_rl:
                        logger.debug('BAT\t%d:\tloss=%.3f,\tml-loss=%.3f,\trl-loss=%.3f,\treward=%.3f', batch_counter,
                                     loss, ml_loss, rl_loss, samples_reward)
                    else:
                        logger.debug('BAT\t%d:\tloss=%.3f,\tml-loss=%.3f,\trl-loss=NA', batch_counter, loss, ml_loss)

                total_loss          += loss
                total_ml_loss       += ml_loss
                total_rl_loss       += rl_loss
                total_samples_award += samples_reward

                batch_counter       += 1

            logger.debug('loss_avg\t=\t%.3f', total_loss / batch_counter)
            logger.debug('ml-loss-avg\t=\t%.3f', total_ml_loss / batch_counter)
            if enable_rl:
                logger.debug('rl-loss_avg\t=\t%.3f,\t reward=%.3f',
                             total_rl_loss / batch_counter, total_samples_award / batch_counter)
            else:
                logger.debug('rl-loss_avg\t=\tNA')

            self.dataLoader.reset()

        # save model
        model_path = FileUtil.get_file_path(conf.get('train:save-model-file'))

        logger.debug('save model into : ' + model_path)

        t.save(self.seq2seq.state_dict(), model_path)

    def evaluate(self):
        self.seq2seq.eval()

        article, _ = self.dataLoader.next()

        summary = self.seq2seq.summarize(article)

        print(summary)

    def get_lr_scheduler(self, optimizer):
        lr_lambda = lambda epoch: self.lr_decay ** (epoch // self.lr_decay_epoch)

        return t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


if __name__ == "__main__":
    train = Train()
    train.run()
    train.evaluate()
