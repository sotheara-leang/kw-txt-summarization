from unittest import TestCase
from main.common.util.file_util import FileUtil


class TestFileUtil(TestCase):

    def test(self):
        print(FileUtil.get_root_dir())
