

class DictUtil:

    """
        Recursive merge dictionaries
    """
    @staticmethod
    def dict_merge(dct: dict, merge_dct: dict):
        for k, v in iter(merge_dct.items()):
            if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict):
                DictUtil.dict_merge(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]
