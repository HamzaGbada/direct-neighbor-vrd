import unittest

from src.dataloader.cord_dataloader import CORD
from src.utils.setup_logger import logger


class TestCordDataLoader(unittest.TestCase):
    def test_json(self):
        train_set = CORD(train=False, download=True)
        logger.debug(f"the cord dataset {train_set.data}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")