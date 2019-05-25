import collections

class DictUtil:

    """
        Recursive merge dictionaries
    """
    @staticmethod
    def merge(dct: dict, merge_dct: dict):
        for k, v in iter(merge_dct.items()):
            if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict):
                DictUtil.merge(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]

    @staticmethod
    def update(dict, key, value):
        _key = key.split(':')[0]
        _next_key = ''.join(key.split(':')[1:])

        for k, v in dict.items():
            if k == _key:
                if isinstance(v, collections.Mapping):
                    DictUtil.update(v, _next_key, value)
                else:
                    if type(value) == type(v):
                        dict[_key] = value
