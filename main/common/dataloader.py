
class DataLoader(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.generator = self.reader()

    def reader(self):
        raise NotImplementedError

    def next_batch(self):
        samples = []
        for i in range(0, self.batch_size):
            try:
                sample = next(self.generator)
            except StopIteration:
                return None

            if sample is None:
                break
            samples.append(sample)

        return samples

    def next(self):
        try:
            return next(self.generator)
        except StopIteration:
            return None

    def reset(self):
        self.generator = self.reader()

