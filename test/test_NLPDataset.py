import pandas as pd
import numpy as np
import unittest
from src import NLPDataset

class TestNLPDataset(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe
        data = {
            'x1': [1, 2, 3, 4, 5],
            'x2': [6, 7, 8, 9, 10],
            'y': [11, 12, 13, 14, 15]
        }
        self.df = pd.DataFrame(data)
        self.x_names = ['x1', 'x2']
        self.y_name = 'y'
        self.dataset = NLPDataset(self.df, self.x_names, self.y_name)

    def test_getDataFrame(self):
        self.assertEqual(self.dataset.getDataFrame().equals(self.df), True)

    def test_getX(self):
        expected_x = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [6, 7, 8, 9, 10]})
        self.assertEqual(self.dataset.getX().equals(expected_x), True)

    def test_getY(self):
        expected_y = pd.DataFrame({'y': [11, 12, 13, 14, 15]})
        self.assertEqual(self.dataset.getY().equals(expected_y), True)

    def test_getFilteredDataFrame(self):
        expected_filtered_df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [6, 7, 8, 9, 10],
            'y': [11, 12, 13, 14, 15]
        })
        self.assertEqual(self.dataset.getFilteredDataFrame().equals(expected_filtered_df), True)

    def test_getXAsNumpyArray(self):
        expected_x_array = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        np.testing.assert_array_equal(self.dataset.getXAsNumpyArray(), expected_x_array)

    def test_getYAsNumpyArray(self):
        expected_y_array = np.array([11, 12, 13, 14, 15])
        np.testing.assert_array_equal(self.dataset.getYAsNumpyArray(), expected_y_array)

    def test_describe(self):
        expected_summary = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [6.0, 7.0, 8.0, 9.0, 10.0],
            'y': [11.0, 12.0, 13.0, 14.0, 15.0]
        }).describe()
        self.assertEqual(self.dataset.describe().equals(expected_summary), True)

    def test_get_shape(self):
        expected_shape = (5, 3)
        self.assertEqual(self.dataset.get_shape(), expected_shape)

    def test_get_columns(self):
        expected_columns = ['x1', 'x2', 'y']
        self.assertEqual(self.dataset.get_columns(), expected_columns)

    def test_modify(self):
        def modify_func(df):
            df['x1'] = df['x1'] * 2
            return df

        self.dataset.modify(modify_func)
        expected_modified_df = pd.DataFrame({
            'x1': [2, 4, 6, 8, 10],
            'x2': [6, 7, 8, 9, 10],
            'y': [11, 12, 13, 14, 15]
        })
        self.assertEqual(self.dataset.getDataFrame().equals(expected_modified_df), True)

    def test_view(self):
        def view_func(df):
            print(df)

        # Redirect stdout to capture the printed output
        import sys
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        self.dataset.view(view_func)
        sys.stdout = sys.__stdout__  # Reset stdout

        expected_output = "   x1  x2   y\n0   1   6  11\n1   2   7  12\n2   3   8  13\n3   4   9  14\n4   5  10  15\n"
        self.assertEqual(captured_output.getvalue(), expected_output)

if __name__ == '__main__':
    unittest.main()