from unittest import TestCase
import os


class TestConfiguration(TestCase):

    def test(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(''))
        print(ROOT_DIR)



