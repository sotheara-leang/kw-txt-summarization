
class DataUtil:

    @staticmethod
    def padding(data: list[any], max_len, pad_id):
        while len(data) < max_len:
            data.append(pad_id)

