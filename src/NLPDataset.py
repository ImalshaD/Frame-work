from __future__ import annotations
from .Dataset import Dataset
import pandas as pd
import numpy as np


class NLPDataset(Dataset[pd.DataFrame]):
    
    @staticmethod
    def fromCSV(path: str) -> NLPDataset:
        dataframe = pd.read_csv(path)
        return NLPDataset(dataframe)

    def getX(self) -> pd.DataFrame:
        """
        Get the input features.

        Returns:
            pd.DataFrame: The input features.
        """
        if not(self._x_names) or not(len(self._x_names)):
            raise ValueError("Input features not set.")
        
        return self._dataframe[self._x_names]
    
    def getY(self) -> pd.DataFrame:
        """
        Get the target variable.

        Returns:
            pd.DataFrame: The target variable.
        """
        if not(self._y_name):
            raise ValueError("Target features not set")
        
        return self._dataframe[[self._y_name]]

    def getFilteredDataFrame(self) -> pd.DataFrame:
        """
        Get the filtered dataframe containing only the input features and the target variable.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        return self._dataframe[self._x_names + [self._y_name]]
    
    def getXAsNumpyArray(self) -> np.ndarray:
        """
        Get the input features as a numpy array.

        Returns:
            np.ndarray: The input features as a numpy array.
        """
        return np.array(self._dataframe[self._x_names])
    
    def getYAsNumpyArray(self) -> np.ndarray:
        """
        Get the target variable as a numpy array.

        Returns:
            np.ndarray: The target variable as a numpy array.
        """
        return np.array(self._dataframe[self._y_name])
    
    def describe(self) -> pd.DataFrame:
        """
        Get a statistical summary of the dataframe.
        
        Returns:
            pd.DataFrame: Statistical summary of the dataframe.
        """
        return self._dataframe.describe()
    
    def get_shape(self) -> tuple:
        """
        Get the shape of the dataframe.
        
        Returns:
            tuple: Shape of the dataframe (rows, columns).
        """
        return self._dataframe.shape
    
    def get_columns(self) -> list:
        """
        Get the list of column names in the dataframe.
        
        Returns:
            list: List of column names.
        """
        return self._dataframe.columns.tolist()
    