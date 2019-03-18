from unittest import TestCase
from main.conf.configuration import Configuration


class TestConfiguration(TestCase):

    def test(self):
        conf = Configuration()
        print(conf.get('training'))



