import os


class FileUtil:

    @staticmethod
    def get_root_dir():
        return os.path.dirname(os.path.abspath(''))

    @staticmethod
    def get_file_path(file: str):
        return os.path.join(FileUtil.get_root_dir(), file)
