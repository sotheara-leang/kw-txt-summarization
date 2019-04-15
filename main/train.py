from rouge import Rouge
import torch.nn as nn
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import math
import os

from main.data.dataloader import *
from main.seq2seq import Seq2Seq
from main.common.batch import *
from main.common.util.file_util import FileUtil
from main.common.glove.vocab import GloveVocab
from main.common.simple_vocab import SimpleVocab
from main.common.glove.embedding import GloveEmbedding


class Train(object):

    def __init__(self):
        self.logger                     = getLogger(self)

        self.hidden_size                = conf.get('hidden-size')
        self.max_enc_steps              = conf.get('max-enc-steps')
        self.max_dec_steps              = conf.get('max-dec-steps')

        self.epoch                      = conf.get('train:epoch')
        self.batch_size                 = conf.get('train:batch-size')
        self.clip_gradient_max_norm     = conf.get('train:clip-gradient-max-norm')
        self.log_batch                  = conf.get('train:log-batch')
        self.log_batch_interval         = conf.get('train:log-batch-interval', -1)
        self.lr                         = conf.get('train:lr')
        self.lr_decay_epoch             = conf.get('train:lr-decay-epoch')
        self.lr_decay                   = conf.get('train:lr-decay')

        self.ml_enable                  = conf.get('train:ml:enable')
        self.ml_forcing_ratio           = conf.get('train:ml:forcing-ratio', 1)
        self.ml_forcing_decay           = conf.get('train:ml:forcing-decay', 0)

        self.rl_enable                  = conf.get('train:rl:enable')
        self.rl_weight                  = conf.get('train:rl:weight')
        self.rl_transit_epoch           = conf.get('train:rl:transit-epoch', -1)
        self.rl_transit_decay           = conf.get('train:rl:transit-decay', 0)

        self.vocab = SimpleVocab(FileUtil.get_file_path(conf.get('train:vocab-file')), conf.get('vocab-size'))
        #self.vocab = GloveVocab(FileUtil.get_file_path(conf.get('train:vocab-file')))

        self.seq2seq = cuda(Seq2Seq(self.vocab))

        #self.seq2seq = cuda(Seq2Seq(self.vocab, GloveEmbedding(FileUtil.get_file_path(conf.get('train:emb-file')))))

        self.batch_initializer = BatchInitializer(self.vocab, self.max_enc_steps, self.max_dec_steps)

        self.data_loader = DataLoader(FileUtil.get_file_path(conf.get('train:article-file')),
                                          FileUtil.get_file_path(conf.get('train:summary-file')),
                                            FileUtil.get_file_path(conf.get('train:keyword-file')), self.batch_size)

        self.optimizer = t.optim.Adam(self.seq2seq.parameters(), lr=self.lr)

        self.criterion = nn.NLLLoss(reduction='none', ignore_index=TK_PADDING['id'])

        self.tb_writer = SummaryWriter(FileUtil.get_file_path(conf.get('train:tb-log-dir')))

    def train_batch(self, batch, epoch_counter):
        self.optimizer.zero_grad()

        rouge       = Rouge()
        dec_input   = cuda(t.tensor([TK_START['id']] * self.batch_size))  # B

        ## encoding input

        x = self.seq2seq.embedding(batch.articles)  # B, L, E

        enc_outputs, (enc_hidden, enc_cell) = self.seq2seq.encoder(x, batch.articles_len)  # (B, L, 2H), (B, 2H), (B, 2H)

        enc_cell = cuda(t.zeros(self.batch_size, 2 * self.hidden_size))

        ## encoding keyword

        kw = self.seq2seq.embedding(batch.keywords)  # B, L, E

        ## ML

        if self.ml_enable:
            output = self.train_ml(enc_outputs, enc_hidden, enc_cell, dec_input, kw, batch, epoch_counter)

            ml_loss = t.sum(output[1], dim=1) / batch.summaries_len.float()
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
            sampling_output = self.train_rl(enc_outputs, enc_hidden, enc_cell, dec_input, kw, batch, True)

            # greedy search
            with t.autograd.no_grad():
                baseline_output = self.train_rl(enc_outputs, enc_hidden, enc_cell, dec_input, kw, batch, False)

            # convert decoded output to string

            sampling_summaries = []
            sampling_outputs = sampling_output[0].tolist()
            for idx, summary in enumerate(sampling_outputs):
                sampling_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

            baseline_summaries = []
            baseline_outputs = baseline_output[0].tolist()
            for idx, summary in enumerate(baseline_outputs):
                baseline_summaries.append(' '.join(self.vocab.ids2words(summary, batch.oovs[idx])))

            reference_summaries = batch.original_summaries

            # calculate rouge score

            sampling_scores = rouge.get_scores(list(sampling_summaries), list(reference_summaries))
            sampling_scores = cuda(t.tensor([score["rouge-l"]["f"] for score in sampling_scores]))

            baseline_scores = rouge.get_scores(list(baseline_summaries), list(reference_summaries))
            baseline_scores = cuda(t.tensor([score["rouge-l"]["f"] for score in baseline_scores]))

            # loss

            sampling_loss = t.sum(sampling_output[1], dim=1) / t.sum(sampling_output[1] != 0, dim=1).float()

            rl_loss = (baseline_scores - sampling_scores) * sampling_loss
            rl_loss = t.mean(rl_loss)

            # reward

            reward = t.mean(sampling_scores - baseline_scores)
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

    def train_ml(self, enc_outputs, dec_hidden, dec_cell, dec_input, kw, batch, epoch_counter):
        y                       = None  # B, T
        loss                    = None  # B, T
        enc_temporal_score      = None  # B, L
        pre_dec_hiddens         = None  # B, T, 2H
        stop_decoding_mask      = cuda(t.zeros(self.batch_size))     # B
        extend_vocab_articles   = batch.extend_vocab_articles        # B, L
        target_y                = batch.summaries
        articles_padding_mask   = batch.articles_padding_mask        # B, L
        max_dec_len             = max(batch.summaries_len)
        max_ovv_len             = max([len(oov) for oov in batch.oovs])

        enc_ctx_vector          = cuda(t.zeros(self.batch_size, 2 * self.hidden_size))

        for i in range(max_dec_len):
            ## decoding
            vocab_dist, dec_hidden, dec_cell, _, _, enc_temporal_score = self.seq2seq.decode(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                articles_padding_mask,
                enc_temporal_score,
                enc_ctx_vector,
                extend_vocab_articles,
                max_ovv_len,
                kw)

            ## loss

            step_loss = self.criterion(t.log(vocab_dist + 1e-10), target_y[:, i])  # B

            for v in step_loss:
                if math.isnan(v) or math.isinf(v):
                    print('>>>> step_loss', step_loss)
                    return

            loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)  # B, L

            ## output

            _, dec_output = t.max(vocab_dist, dim=1)

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            ## masking decoding

            stop_decoding_mask[(stop_decoding_mask == 0) + (dec_output == TK_STOP['id']) == 2] = 1

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

    def train_rl(self, enc_outputs, dec_hidden, dec_cell, dec_input, kw, batch, sampling):
        y                       = None  # B, T
        loss                    = None  # B, T
        enc_temporal_score      = None  # B, L
        pre_dec_hiddens         = None  # B, T, 2H
        stop_decoding_mask      = cuda(t.zeros(self.batch_size))    # B
        articles_padding_mask   = batch.articles_padding_mask       # B, L
        extend_vocab_articles   = batch.extend_vocab_articles       # B, L
        max_ovv_len             = max([len(oov) for oov in batch.oovs])

        enc_ctx_vector = cuda(t.zeros(self.batch_size, 2 * self.hidden_size))

        for i in range(self.max_dec_steps):
            ## decoding
            vocab_dist, dec_hidden, dec_cell, _, _, enc_temporal_score = self.seq2seq.decode(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                articles_padding_mask,
                enc_temporal_score,
                enc_ctx_vector,
                extend_vocab_articles,
                max_ovv_len,
                kw)

            ## sampling
            if sampling:
                sampling_dist = Categorical(vocab_dist)
                dec_output = sampling_dist.sample()

                step_loss = sampling_dist.log_prob(dec_output)

                loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)  # B, L
            else:
                ## greedy search
                _, dec_output = t.max(vocab_dist, dim=1)

            ## y

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            ## masking decoding

            stop_decoding_mask[(stop_decoding_mask == 0) + (dec_output == TK_STOP['id']) == 2] = 1

            if len(stop_decoding_mask[stop_decoding_mask == 1]) == len(stop_decoding_mask):
                break

            dec_input = dec_output

            ## if next decoder input is oov, change it to UNKNOWN_TOKEN

            is_oov = (dec_input >= self.vocab.size()).long()

            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN['id']

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

        return y, loss

    def train(self):
        self.seq2seq.train()

        self.logger.debug('>>> training:')

        total_batch_counter = 0

        criterion_scheduler = t.optim.lr_scheduler.StepLR(self.optimizer, self.lr_decay_epoch, self.lr_decay)

        for i in range(self.epoch):

            self.logger.debug('========================= epoch %i/%i =========================', i + 1, self.epoch)

            batch_counter       = 0

            total_loss          = 0
            total_ml_loss       = 0
            total_rl_loss       = 0
            total_samples_award = 0

            criterion_scheduler.step()

            while True:

                # get next batch
                batch = self.data_loader.next_batch()

                if batch is None:
                    break

                # init batch
                batch = self.batch_initializer.init(batch)

                # feed batch to model
                loss, ml_loss, rl_loss, samples_reward, enable_rl = self.train_batch(batch, i+1)

                if self.log_batch:

                    if self.log_batch_interval <= 0 or (batch_counter + 1) % self.log_batch_interval == 0:

                        # log to tensorboard
                        self.tb_writer.add_scalar('Batch_Train/Loss', loss, total_batch_counter + 1)
                        self.tb_writer.add_scalar('Batch_Train/ML-Loss', ml_loss, total_batch_counter + 1)
                        self.tb_writer.add_scalar('Batch_Train/RL-Loss', rl_loss, total_batch_counter + 1)

                        if enable_rl:
                            self.logger.debug('EP\t%d,\tBAT\t%d:\tloss=%.3f,\tml-loss=%.3f,\trl-loss=%.3f,\treward=%.3f', i + 1, batch_counter + 1,
                                         loss, ml_loss, rl_loss, samples_reward)
                        else:
                            self.logger.debug('EP\t%d,\tBAT\t%d:\tloss=%.3f,\tml-loss=%.3f,\trl-loss=NA', i + 1, batch_counter + 1, loss, ml_loss)

                total_loss          += loss
                total_ml_loss       += ml_loss
                total_rl_loss       += rl_loss
                total_samples_award += samples_reward

                batch_counter       += 1
                total_batch_counter += 1

            epoch_loss = total_loss / batch_counter
            epoch_ml_loss = total_ml_loss / batch_counter
            epoch_rl_loss = total_rl_loss / batch_counter
            epoch_samples_award = total_samples_award / batch_counter

            # log to tensorboard
            self.tb_writer.add_scalar('Epoch_Train/Loss', loss, i + 1)
            self.tb_writer.add_scalar('Epoch_Train/ML-Loss', ml_loss, i + 1)
            self.tb_writer.add_scalar('Epoch_Train/RL-Loss', rl_loss, i + 1)

            self.logger.debug('loss_avg\t=\t%.3f', epoch_loss)
            self.logger.debug('ml-loss-avg\t=\t%.3f', epoch_ml_loss)
            if enable_rl:
                self.logger.debug('rl-loss_avg\t=\t%.3f,\t reward=%.3f', epoch_rl_loss, epoch_samples_award)
            else:
                self.logger.debug('rl-loss_avg\t=\tNA')

            # reload data set
            self.data_loader.reset()

        return i, epoch_loss

    def evaluate(self):
        is_enable = conf.get('train:eval', False)
        if is_enable is False:
            return

        self.logger.debug('>>> evaluation:')

        self.seq2seq.eval()

        rouge = Rouge()
        scores = []

        while True:
            batch = self.data_loader.next_batch()

            if batch is None:
                break

            for sample in batch:
                article = sample[0]
                ref_summary = sample[1]

                gen_summary = self.seq2seq.summarize(article)

                rouge_score = rouge.get_scores(gen_summary, ref_summary)[0]

                scores.append(rouge_score["rouge-l"]["f"])

        avg_score = sum(scores) / len(scores)

        self.logger.debug('examples: %d', len(scores))
        self.logger.debug('avg rouge-l score: %.3f', avg_score)

    def load_model(self):
        model_file = conf.get('train:model-file')
        if model_file is None:
            return
        model_file = FileUtil.get_file_path(model_file)

        if os.path.isfile(model_file):
            self.logger.debug('>>> load pre-trained model from: %s', model_file)

            checkpoint = t.load(model_file)

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            self.seq2seq.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.debug('epoch: %s', str(epoch + 1))
            self.logger.debug('loss: %s', str(loss.item()))
        else:
            self.logger.warning('>>> cannot load pre-trained model - file not exist: %s', model_file)

    def save_model(self, args):
        model_file = conf.get('train:save-model-file')
        if not model_file:
            return

        self.logger.debug('>>> save model into: ' + model_file)

        t.save({
            'epoch': args['epoch'],
            'loss': args['loss'],
            'model_state_dict': self.seq2seq.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, FileUtil.get_file_path(model_file))

    def run(self):
        # display configuration
        self.logger.debug('>>> configuration: \n' + conf.dump().strip())

        # load pre-trained model`
        self.load_model()

        # train
        epoch, loss = self.train()

        # evaluate
        self.evaluate()

        # save model
        self.save_model({'epoch': epoch, 'loss': loss})


if __name__ == "__main__":
    train = Train()
    train.run()
