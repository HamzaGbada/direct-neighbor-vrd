import unittest
import matplotlib.pyplot as plt
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.utils.setup_logger import logger


class TestDataLoader(unittest.TestCase):
    def test_cord(self):
        train_set = CORD(train=False, download=True)
        logger.debug(f"the cord dataset {train_set.data}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")
    def test_wildreceipt(self):
        train_set = WILDRECEIPT(train=False, download=True)
        # logger.debug(f"the cord dataset {train_set.data}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")

    def test_sroie(self):
        train_set = SROIE(train=True)
        # nbr_of_node = train_set.data[0][1]['boxes'].shape
        # logger.debug(f"The shape of bbox in the first doc Dataset: \n{nbr_of_node}")
        # logger.debug(f"The shape of bbox in the first doc Dataset: \n{len(train_set.data[0][1]['boxes'])}")
        # logger.debug(f"The bbox in the first doc Dataset: \n{train_set.data[0]}")
        # # logger.debug(f"The bbox in the first doc Dataset: \n{train_set[55]}")
        # plt.imshow(train_set[0][0].permute(1, 2, 0))
        # plt.show()
        # # logger.debug(f"The size text_unit_list for all dataset: {len(train_set.text_units)}")
        # logger.debug(f"text_unit_list for all dataset: {train_set.text_units}")
        # j = 0
        # for i in train_set.data:
        #     j+=1
        logger.debug(f"The sroie dataset: {train_set.data}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")
