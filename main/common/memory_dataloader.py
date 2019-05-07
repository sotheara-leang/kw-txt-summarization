from main.common.dataloader import DataLoader


class MemoryDataLoader(DataLoader):

    def __init__(self, batch_size):
        super(MemoryDataLoader, self).__init__(batch_size)

        self.data = self.read_all()

        self.num_batch = int(len(self.data) / self.batch_size) + (1 if len(self.data) % self.batch_size > 0 else 0)

    def reader(self):
        raise NotImplementedError

    def len(self):
        return len(self.data)

    def get_num_batch(self):
        return self.num_batch

    def get(self, idx):
        if idx < 0 or idx > self.len():
            return None
        return self.data[idx]

    def get_batch(self, batch_idx):
        if batch_idx < 0 or batch_idx >= self.num_batch:
            return None

        start_idx = self.batch_size * batch_idx

        batch = []
        for i in range(self.batch_size):
            batch.append(self.get(start_idx + i))

        return batch
