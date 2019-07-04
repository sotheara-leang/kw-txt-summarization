import argparse
import datetime
import time

import torch.nn as nn
from rouge import Rouge
from tensorboardX import SummaryWriter
from torch import autograd
from torch.distributions import Categorical

from main.common.batch import *
from main.common.glove.embedding import GloveEmbedding
from main.common.simple_vocab import SimpleVocab
from main.common.util.file_util import FileUtil
from main.data.cnn_dataloader import *
from main.seq2seq import Seq2Seq


class Train(object):

    def __init__(self):
        self.logger = logger(self)

        self.enc_hidden_size        = conf('enc-hidden-size')
        self.dec_hidden_size        = conf('dec-hidden-size')

        self.max_enc_steps          = conf('max-enc-steps')
        self.max_dec_steps          = conf('max-dec-steps')

        self.epoch                  = conf('train:epoch')
        self.clip_gradient_max_norm = conf('train:clip-gradient-max-norm')
        self.log_batch              = conf('train:log-batch')
        self.log_batch_interval     = conf('train:log-batch-interval', -1)
        self.lr                     = conf('train:lr')
        self.lr_decay_epoch         = conf('train:lr-decay-epoch')
        self.lr_decay               = conf('train:lr-decay')

        self.ml_enable              = conf('train:ml:enable', True)
        self.ml_forcing_ratio       = conf('train:ml:forcing-ratio', 1)
        self.ml_forcing_decay       = conf('train:ml:forcing-decay', 0)

        self.rl_enable              = conf('train:rl:enable')
        self.rl_weight              = conf('train:rl:weight')
        self.rl_transit_epoch       = conf('train:rl:transit-epoch', -1)
        self.rl_transit_decay       = conf('train:rl:transit-decay', 0)

        self.save_model_per_epoch   = conf('train:save-model-per-epoch')

        self.tb_log_batch           = conf('train:tb:log-batch')

        # tensorboard
        self.tb_writer = None
        if conf('train:tb:enable') is True:
            tb_log_dir = conf('train:tb:log-dir')
            if tb_log_dir is not None:
                self.tb_writer = SummaryWriter(FileUtil.get_file_path(tb_log_dir))

        self.vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        embedding = GloveEmbedding(FileUtil.get_file_path(conf('emb-file')), self.vocab) if conf('emb-file') is not None else None

        self.seq2seq = cuda(Seq2Seq(self.vocab, embedding))

        self.batch_initializer = BatchInitializer(self.vocab, self.max_enc_steps, self.max_dec_steps, conf('pointer-generator'))

        self.data_loader = CNNDataLoader(FileUtil.get_file_path(conf('train:article-file')),
                                         FileUtil.get_file_path(conf('train:summary-file')),
                                         FileUtil.get_file_path(conf('train:keyword-file')),
                                         conf('train:batch-size'))

        self.optimizer = t.optim.Adam(self.seq2seq.parameters(), lr=self.lr)

        self.criterion = nn.NLLLoss(reduction='none', ignore_index=TK_PADDING['id'])

    def train_batch(self, batch, epoch_counter):
        start_time = time.time()

        ## encoding input

        x = self.seq2seq.embedding(batch.articles)

        enc_outputs, (enc_hidden_n, enc_cell_n) = self.seq2seq.encoder(x, batch.articles_len)

        dec_hidden, dec_cell = self.seq2seq.reduce_encoder(enc_hidden_n, enc_cell_n)

        ## encoding keyword

        kw = self.seq2seq.kw_encoder(batch.keywords)  # B, L, E

        ## ML

        if self.ml_enable:
            ml_loss = self.train_ml(enc_outputs, dec_hidden, dec_cell, kw, batch, epoch_counter)
            ml_loss = t.mean(ml_loss)
        else:
            ml_loss = cuda(t.zeros(1))

        ## RL

        rl_enable = self.rl_enable
        if rl_enable and self.ml_enable is True:
            if self.rl_transit_epoch > 0:
                rl_enable = epoch_counter >= self.rl_transit_epoch
            else:
                rl_enable = max(0, 1 - self.rl_transit_decay * epoch_counter) == 0

        if rl_enable:
            # sampling
            sampling_output = self.train_rl(enc_outputs, dec_hidden, dec_cell, kw, batch, True)

            # greedy search
            with t.autograd.no_grad():
                baseline_output = self.train_rl(enc_outputs, dec_hidden, dec_cell, kw, batch, False)

            # convert decoded output to string

            sampling_summaries = []
            sampling_outputs = sampling_output[0].tolist()
            for idx, summary in enumerate(sampling_outputs):
                try:
                    stop_idx = summary.index(TK_STOP['id'])
                    summary = summary[:stop_idx]
                except ValueError:
                    pass

                summary = ' '.join(self.vocab.ids2words(summary, batch.oovs[idx]))

                sampling_summaries.append(summary)

            baseline_summaries = []
            baseline_outputs = baseline_output[0].tolist()
            for idx, summary in enumerate(baseline_outputs):
                try:
                    stop_idx = summary.index(TK_STOP['id'])
                    summary = summary[:stop_idx]
                except ValueError:
                    pass

                summary = ' '.join(self.vocab.ids2words(summary, batch.oovs[idx]))

                baseline_summaries.append(summary)

            reference_summaries = batch.original_summaries

            # calculate rouge score

            sampling_scores = self.get_reward(sampling_summaries, reference_summaries)
            sampling_scores = cuda(sampling_scores)

            baseline_scores = self.get_reward(baseline_summaries, reference_summaries)
            baseline_scores = cuda(baseline_scores)

            # loss

            sampling_log_prob = sampling_output[1]

            rl_loss = (baseline_scores - sampling_scores) * sampling_log_prob
            rl_loss = t.mean(rl_loss)

            # reward

            reward = t.mean(sampling_scores).item()

            rl_weight = self.rl_weight if self.ml_enable else 1
        else:
            rl_loss = cuda(t.zeros(1))
            reward = 0
            rl_weight = 0

        ## total loss

        loss = (1 - rl_weight) * ml_loss + rl_weight * rl_loss

        self.optimizer.zero_grad()

        loss.backward()

        if self.clip_gradient_max_norm is not None:
            nn.utils.clip_grad_norm_(self.seq2seq.parameters(), self.clip_gradient_max_norm)

        self.optimizer.step()

        t.cuda.empty_cache()

        loss = loss.detach().item()
        ml_loss = ml_loss.detach().item()
        rl_loss = rl_loss.detach().item()

        time_spent = time.time() - start_time

        return loss, ml_loss, rl_loss, reward, rl_enable, time_spent

    def train_ml(self, enc_outputs, dec_hidden, dec_cell, kw, batch, epoch_counter):
        loss                = None
        enc_temporal_score  = None
        pre_dec_hiddens     = None
        extend_vocab_x      = batch.extend_vocab_articles
        enc_padding_mask    = batch.articles_padding_mask
        target_y            = batch.summaries
        max_dec_len         = max(batch.summaries_len)
        max_ovv_len         = max([len(oov) for oov in batch.oovs])
        dec_input           = batch.summaries[:, 0]
        enc_ctx_vector      = cuda(t.zeros(batch.size, 2 * self.enc_hidden_size))

        for i in range(1, max_dec_len):
            dec_input = self.seq2seq.embedding(dec_input)

            ## decoding
            vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, _, enc_temporal_score, _, _ = self.seq2seq.decoder(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                enc_padding_mask,
                enc_temporal_score,
                enc_ctx_vector,
                extend_vocab_x,
                max_ovv_len,
                kw)

            ## loss

            step_loss = self.criterion(t.log(vocab_dist + 1e-20), target_y[:, i])

            loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)

            ## output

            dec_output = t.multinomial(vocab_dist, 1).squeeze(1).detach()

            ## teacher forcing

            forcing_ratio = max(0, self.ml_forcing_ratio - self.ml_forcing_decay * epoch_counter)

            use_ground_truth = cuda((t.rand(batch.size) < forcing_ratio).long())

            dec_input = use_ground_truth * target_y[:, i] + (1 - use_ground_truth) * dec_output

            ## if next decoder input is oov, change it to TK_UNKNOWN

            is_oov = (dec_input >= self.vocab.size()).long()

            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN['id']

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

        loss = t.sum(loss, dim=1) / batch.summaries_len.float()

        return loss

    def train_rl(self, enc_outputs, dec_hidden, dec_cell, kw, batch, sampling):
        y                       = None
        log_prob                = None
        enc_temporal_score      = None
        pre_dec_hiddens         = None
        enc_padding_mask        = batch.articles_padding_mask
        extend_vocab_x          = batch.extend_vocab_articles
        max_ovv_len             = max([len(vocab) for vocab in batch.oovs])
        dec_input               = batch.summaries[:, 0]
        enc_ctx_vector          = cuda(t.zeros(batch.size, 2 * self.enc_hidden_size))
        stop_decoding_mask      = cuda(t.zeros(batch.size))
        decoding_padding_mask   = []

        for i in range(1, self.max_dec_steps):
            dec_input = self.seq2seq.embedding(dec_input)

            ## decoding
            vocab_dist, dec_hidden, dec_cell, _, _, enc_temporal_score, _, _ = self.seq2seq.decoder(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                enc_padding_mask,
                enc_temporal_score,
                enc_ctx_vector,
                extend_vocab_x,
                max_ovv_len,
                kw)

            ## sampling
            if sampling:
                sampling_dist = Categorical(vocab_dist)
                dec_output = sampling_dist.sample()

                step_log_prob = sampling_dist.log_prob(dec_output)

                log_prob = step_log_prob.unsqueeze(1) if log_prob is None else t.cat([log_prob, step_log_prob.unsqueeze(1)], dim=1)
            else:
                ## greedy search
                _, dec_output = t.max(vocab_dist, dim=1)

            ## output

            dec_output = dec_output.detach()

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            ## stop decoding mask

            decoding_padding_mask.append(1 - stop_decoding_mask)

            stop_decoding_mask[dec_output == TK_STOP['id']] = 1

            if len(stop_decoding_mask[stop_decoding_mask == 1]) == len(stop_decoding_mask):
                break

            ## if next decoder input is oov, change it to TK_UNKNOWN

            dec_input = dec_output

            is_oov = (dec_input >= self.vocab.size()).long()

            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN['id']

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

        # masking padding - not considering sampled words with padding mask = 0

        if sampling:
            decoding_padding_mask = t.stack(decoding_padding_mask, dim=1)

            log_prob = log_prob * decoding_padding_mask

            sampling_lens = t.sum(decoding_padding_mask, dim=1)

            log_prob = t.sum(log_prob, dim=1) / sampling_lens

        return y, log_prob

    def train(self, start_epoch):
        self.seq2seq.train()

        self.logger.debug('>>> training:')

        total_batch_counter = 0
        train_time = time.time()

        criterion_scheduler = t.optim.lr_scheduler.StepLR(self.optimizer, self.lr_decay_epoch, self.lr_decay)

        for i in range(start_epoch, self.epoch):
            self.logger.debug('========================= epoch %i/%i =========================', i + 1, self.epoch)

            batch_counter = 0

            total_loss = 0
            total_ml_loss = 0
            total_rl_loss = 0
            total_samples_award = 0

            epoch_time_spent = 0

            criterion_scheduler.step()

            while True:
                # get next batch
                batch = self.data_loader.next_batch()

                if batch is None:
                    break

                # init batch
                batch = self.batch_initializer.init(batch)

                # feed batch to model
                loss, ml_loss, rl_loss, samples_reward, enable_rl, time_spent = self.train_batch(batch, i + 1)

                epoch_time_spent += time_spent

                # log to tensorboard
                if self.tb_writer is not None and self.tb_log_batch is True:
                    self.tb_writer.add_scalar('Batch_Train/Loss', loss, total_batch_counter + 1)
                    self.tb_writer.add_scalar('Batch_Train/ML-Loss', ml_loss, total_batch_counter + 1)
                    self.tb_writer.add_scalar('Batch_Train/RL-Loss', rl_loss, total_batch_counter + 1)

                if self.log_batch and self.log_batch_interval <= 0 or (batch_counter + 1) % self.log_batch_interval == 0:
                    self.logger.debug(
                        'EP\t%d,\tBAT\t%d:\tloss=%.3f,\tml-loss=%.3f,\trl-loss=%.3f,\treward=%.3f,\ttime=%s',
                        i + 1,
                        batch_counter + 1,
                        loss, ml_loss, rl_loss, samples_reward, str(datetime.timedelta(seconds=time_spent)))

                total_loss += loss
                total_ml_loss += ml_loss
                total_rl_loss += rl_loss
                total_samples_award += samples_reward

                batch_counter += 1
                total_batch_counter += 1

            epoch_loss = total_loss / batch_counter
            epoch_ml_loss = total_ml_loss / batch_counter
            epoch_rl_loss = total_rl_loss / batch_counter
            epoch_samples_award = total_samples_award / batch_counter

            # log to tensorboard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('Epoch_Train/Loss', epoch_loss, i + 1)
                self.tb_writer.add_scalar('Epoch_Train/ML-Loss', epoch_ml_loss, i + 1)
                self.tb_writer.add_scalar('Epoch_Train/RL-Loss', epoch_rl_loss, i + 1)

            self.logger.debug('loss_avg\t=\t%.3f', epoch_loss)
            self.logger.debug('ml-loss-avg\t=\t%.3f', epoch_ml_loss)
            self.logger.debug('rl-loss_avg\t=\t%.3f,\t reward=%.3f', epoch_rl_loss, epoch_samples_award)
            self.logger.debug('rl-loss_avg\t=\tNA')
            self.logger.debug('time\t=\t%s', str(datetime.timedelta(seconds=epoch_time_spent)))

            # reload data set
            self.data_loader.reset()

            # save model
            if i == self.epoch - 1 or (self.save_model_per_epoch is not None and (i + 1) % self.save_model_per_epoch == 0):
                self.save_model({'epoch': i, 'loss': epoch_loss})

        train_time = time.time() - train_time

        self.logger.debug('time\t=\t%s', str(datetime.timedelta(seconds=train_time)))

    def evaluate(self):
        is_enable = conf('train:eval', False)
        if is_enable is False:
            return

        self.logger.debug('>>> evaluation:')

        self.seq2seq.eval()

        rouge = Rouge()
        total_scores_1 = []
        total_scores_2 = []
        total_scores_l = []
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
                try:
                    stop_idx = summary.index(TK_STOP['id'])
                    summary = summary[:stop_idx]
                except ValueError:
                    pass

                summary = ' '.join(self.vocab.ids2words(summary, batch.oovs[idx]))

                gen_summaries.append(summary)

            reference_summaries = batch.original_summaries

            # calculate rouge score

            avg_score = rouge.get_scores(gen_summaries, reference_summaries, avg=True)
            avg_score_1 = avg_score["rouge-1"]["f"]
            avg_score_2 = avg_score["rouge-2"]["f"]
            avg_score_l = avg_score["rouge-l"]["f"]

            eval_time = time.time() - eval_time

            if self.log_batch_interval <= 0 or (batch_counter + 1) % self.log_batch_interval == 0:
                self.logger.debug('BAT\t%d:\t\trouge-1=%.3f\t\trouge-2=%.3f\t\trouge-l=%.3f\t\ttime=%s', batch_counter + 1,
                                  avg_score_1, avg_score_2, avg_score_l,
                                  str(datetime.timedelta(seconds=eval_time)))

            total_scores_1.append(avg_score_1)
            total_scores_2.append(avg_score_2)
            total_scores_l.append(avg_score_l)

            batch_counter += 1
            example_counter += batch.size

        avg_score_1 = sum(total_scores_1) / len(total_scores_1)
        avg_score_2 = sum(total_scores_2) / len(total_scores_2)
        avg_score_l = sum(total_scores_l) / len(total_scores_l)

        total_eval_time = time.time() - total_eval_time

        self.logger.debug('examples: %d', example_counter)
        self.logger.debug('avg rouge-1: %f', avg_score_1)
        self.logger.debug('avg rouge-2: %f', avg_score_2)
        self.logger.debug('avg rouge-l: %f', avg_score_l)

        self.logger.debug('time\t:\t%s', str(datetime.timedelta(seconds=total_eval_time)))

    def load_model(self):
        epoch = 0

        model_file = conf('train:load-model-file')
        if model_file is None:
            return epoch

        model_file = FileUtil.get_file_path(model_file)

        if os.path.isfile(model_file):
            self.logger.debug('>>> load pre-trained model from: %s', model_file)

            if conf('device') == 'cpu':
                checkpoint = t.load(model_file, map_location='cpu')
            else:
                checkpoint = t.load(model_file)

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            self.seq2seq.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.debug('epoch: %s', str(epoch))
            self.logger.debug('loss: %s', str(loss))
        else:
            raise Exception('>>> error loading model - file not exist: %s' % model_file)

        return epoch

    def save_model(self, param):
        model_file = conf('train:save-model-file')
        if not model_file:
            return

        model_file = FileUtil.get_file_path(model_file)

        file_dir, _ = os.path.split(model_file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        dot = model_file.rfind('.')
        if dot != -1:
            model_file = model_file[:dot] + '-' + str(param['epoch'] + 1) + model_file[dot:]
        else:
            model_file = model_file + '-' + str(param['epoch'] + 1)

        self.logger.debug('>>> save model into: ' + model_file)

        self.logger.debug('epoch: %s', str(param['epoch'] + 1))
        self.logger.debug('loss: %s', str(param['loss']))

        t.save({
            'epoch': param['epoch'] + 1,
            'loss': param['loss'],
            'model_state_dict': self.seq2seq.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, FileUtil.get_file_path(model_file))

    def run(self):
        try:
            # display configuration
            self.logger.debug('>>> configuration: \n' + conf().dump().strip())

            # load pre-trained model
            current_epoch = self.load_model()

            #
            total_params = sum(p.numel() for p in self.seq2seq.parameters() if p.requires_grad)
            self.logger.debug('>>> total parameters: %d', total_params)

            # train
            #with autograd.detect_anomaly():
            self.train(current_epoch)

            # evaluate
            self.evaluate()
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise e

    def get_reward(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            self.logger.error("Rouge failed for multi sentence evaluation..")

            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i], original_sents[i])
                except Exception:
                    self.logger.error("decoded_sents: %s", decoded_sents[i])
                    self.logger.error("original_sents: %s", original_sents[i])

                    score = [{"rouge-l": {"f": 0.0}}]

                scores.append(score[0])

        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        rouge_l_f1 = t.Tensor(rouge_l_f1)

        return rouge_l_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, default='main/conf/train/config.yml')

    args = parser.parse_args()

    AppContext(args.conf_file)

    train = Train()
    train.run()
