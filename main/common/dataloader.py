from main.common.common import *


class DataLoader(object):

    def __init__(self, batch_size):
        self.logger = getLogger(self)

        self.batch_size = batch_size
        self.generator = self.reader()

    def reader(self):
        raise NotImplementedError

    def next_batch(self):
        samples = []
        for i in range(0, self.batch_size):
            sample = None
            try:
                sample = next(self.generator)
            except Exception:
                pass

            if sample is None:
                break
            samples.append(sample)

        return samples if len(samples) > 0 else None

    def next(self):
        try:
            return next(self.generator)
        except StopIteration:
            return None
        except Exception as e:
            raise e

    def read_all(self):
        samples = []
        while True:
            sample = self.next()
            if sample is None:
                break

            samples.append(sample)

        self.reset()

        return samples

    def reset(self):
        self.generator = self.reader()


