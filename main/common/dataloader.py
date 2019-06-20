import queue as Queue
from random import shuffle

from main.common.common import *

class DataLoader(object):

    EXAMPLE_QUEUE_MAX = 2000

    def __init__(self, batch_size, mode='train'):    # eval|train
        self.logger = logger(self)

        self.batch_size = batch_size
        self.mode = mode
        self.generator = self.reader()

        self.example_queue = Queue.Queue(DataLoader.EXAMPLE_QUEUE_MAX)

    def reader(self):
        raise NotImplementedError

    def next_batch(self):
        samples = []
        for i in range(self.batch_size):
            sample = None
            try:
                sample = self.next()
            except Exception:
                pass

            if sample is None:
                break
            samples.append(sample)

        return samples if len(samples) > 0 else None

    def next(self):
        if self.mode == 'train':
            if self.example_queue.empty() or self.example_queue.qsize() < self.batch_size:
                examples = []

                examples_len = DataLoader.EXAMPLE_QUEUE_MAX - self.example_queue.qsize()
                for i in range(examples_len):
                    try:
                        example = next(self.generator)
                    except StopIteration:
                        example = None

                    if example is None:
                        break

                    examples.append(example)

                if self.mode == 'train':
                    shuffle(examples)

                for example in examples:
                    self.example_queue.put(example)

            example = None if self.example_queue.qsize() == 0 else self.example_queue.get(block=False)
        else:
            example = next(self.generator)

        return example

    def read_all(self):
        self.reset()

        samples = []
        while True:
            try:
                sample = next(self.generator)
                if sample is None:
                    break

                samples.append(sample)
            except Exception:
                pass

        self.reset()

        return samples

    def reset(self):
        self.generator = self.reader()


