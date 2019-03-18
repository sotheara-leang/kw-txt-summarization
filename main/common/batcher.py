# Most codes are from https://github.com/rohithreddy024/Text-Summarizer-Pytorch/blob/master/data_util/batcher.py

from queue import Queue
import time

from random import shuffle
from threading import Thread

from main.common.common import *
from main.common.sample import Sample
from main.common.batch import Batch


class Batcher(object):

    BATCH_QUEUE_MAX = 1000  # max number of batches the batch_queue can hold

    def __init__(self, article_path, summary_path, vocab, mode, batch_size, single_pass):
        self._article_path = article_path
        self._summary_path = summary_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size

        # Initialize a queue of Batches waiting to be used, and a queue of Samples waiting to be batched
        self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
        self._sample_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_sample_q_threads = 1      # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1       # just one thread to batch sample
            self._bucketing_cache_size = 1      # only load one batch's worth of sample before bucketing; this essentially means no bucketing
            self._finished_reading = False      # this will tell us when we're finished reading the dataset
        else:
            self._num_sample_q_threads = 1      # 16 # num threads to fill sample queue
            self._num_batch_q_threads = 1       # 4  # num threads to fill batch queue
            self._bucketing_cache_size = 1      # 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._sample_q_threads = []
        for _ in range(self._num_sample_q_threads):
            self._sample_q_threads.append(Thread(target=self.fill_sample_queue))
            self._sample_q_threads[-1].daemon = True
            self._sample_q_threads[-1].start()

        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads should not run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_sample_queue(self):
        sample_generator = self.sample_generator(self._article_path, self._summary_path, self._single_pass)

        while True:
            try:
                (article, summary) = next(sample_generator)

            except StopIteration:  # if there are no more examples:
                logging.info("The sample generator for this sample queue filling thread has exhausted data.")

                if self._single_pass:
                    logging.info("single_pass mode is on, so we've finished reading datafile. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the sample generator is out of data; error.")

            sample = Sample(article, summary, self._vocab)

            self._sample_queue.put(sample)  # place the Example in the sample queue.

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single sample repeated in the batch
                ex = self._sample_queue.get()
                b = [ex for _ in range(self.batch_size)]

                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._sample_queue.get())

                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            logging.info('Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._sample_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._sample_q_threads):
                if not t.is_alive():  # if the thread is dead
                    logging.error('Found sample queue thread dead. Restarting.')

                    new_t = Thread(target=self.fill_sample_queue)
                    self._sample_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    logging.error('Found batch queue thread dead. Restarting.')

                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def sample_generator(self, article_path, summary_path, single_pass):
        while True:
            with open(article_path, 'r') as art_reader, open(summary_path, 'r') as sum_reader:
                while True:
                    article = next(art_reader)
                    summary = next(sum_reader)

                    if article == '' or summary == '':
                        break

                    yield Sample(article, summary, self._vocab)

                if single_pass:
                    break
