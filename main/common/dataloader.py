
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
            except Exception:
                return None

            if sample is None:
                break
            samples.append(sample)

        return samples

    def next(self):
        try:
            return next(self.generator)
        except Exception as e:
            return None

    def read_all(self):
        samples = []
        while True:
            sample = self.next()
            if sample is None:
                self.reset()
                break

            samples.append(sample)
        return samples

    def reset(self):
        self.generator = self.reader()


