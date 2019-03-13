from unittest import TestCase
from main.util.file_util import FileUtil


class TestFileUtil(TestCase):

    def test(self):
        print(FileUtil.get_root_dir())
