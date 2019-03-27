import os


class FileUtil:

    @staticmethod
    def get_proj_dir():
        return os.environ['PROJ_HOME']

    @staticmethod
    def get_file_path(file: str):
        return os.path.join(FileUtil.get_proj_dir(), file)
