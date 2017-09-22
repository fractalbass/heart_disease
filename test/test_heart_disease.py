import unittest
from heart_disease import HeartDisease
import numpy as np


class test_heart_disease(unittest.TestCase):

    def test_load_data(self):
        hd = HeartDisease()
        datafile = hd.load_dataset("./unit_testing_data.csv")
        self.assertEqual(datafile.shape, (10, 14))
        self.assertEqual(hd.max_rows, 10)
        self.assertEqual(hd.max_rows, 14)

    #  Make sure that the data falls between -1 and 1 for all columns.
    def test_preprocess_data(self):
        hd = HeartDisease()
        datafile = hd.load_dataset("./unit_testing_data.csv")
        self.assertEqual(datafile.shape, (10, 14))
        data = hd.preprocess(datafile)
        (r,c) = data.shape
        self.assertEqual(r, 10)
        self.assertEqual(c, 14)

        for x in range(0,c-1):
            mx = np.max(data[0:10, x])
            mn = np.min(data[0:10, x])
            print("{0} -> {1}, {2}".format(data[0:10,x], mn, mx))
            self.assertTrue(mx <= 1.1)
            self.assertTrue(mn >= -1.1)

    def test_split_dataset(self):
        hd = HeartDisease()
        datafile = hd.load_dataset("./unit_testing_data.csv")
        self.assertTrue(datafile.shape==(10, 14))
        X, Y = hd.split_dataset(datafile)
        print("Shape of X={0}   Shape of Y={1}".format(X.shape, Y.shape))

        self.assertTrue(X.shape == (10, 13))

        # The slice just drops this down to an array of a single dimension.
        self.assertTrue(Y.shape == (10, ))

    def test_create_model(self):
        hd = HeartDisease()
        hd.create_model()
        self.assertTrue(hd.model is not None)

    def test_get_category(self):
        hd = HeartDisease()
        # Test 5 class category
        self.assertTrue(hd.get_five_class_category(1.1) == 1)
        self.assertTrue(hd.get_five_class_category(.6) == 0.5)
        self.assertTrue(hd.get_five_class_category(.2) == 0)
        self.assertTrue(hd.get_five_class_category(-.52) == -0.5)
        self.assertTrue(hd.get_five_class_category(-.8) == -1)

        # Test 2 class category
        self.assertTrue(hd.get_binary_category(.75) == 1)
        self.assertTrue(hd.get_binary_category(.25) == 0)

