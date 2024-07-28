from typing import TypeVar, Generic, Callable
from abc import ABC, abstractmethod

T = TypeVar("T")

class Dataset(ABC, Generic[T]):
    
    def __init__(self, dataset: T, x_names: list[str] = None, y_name: str = None) -> None:
        """
        Initialize the NLPDataset object.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing the data.
            x_names (list[str]): The list of column names to be used as input features.
            y_name (str): The column name to be used as the target variable.
        """
        super().__init__()
        self._dataframe : T = dataset
        self._x_names : list[str] = x_names
        self._y_name : str = y_name

    def _checkDataFrame(self):
        """
        Checks if the dataset has been set.

        Raises:
            ValueError: If the dataset has not been set.
        """
        if not(self._dataframe):
            raise ValueError("Dataset not set")
    
    def check_x_names(self):
        """
        Check if the x_names attribute is set.

        Raises:
            ValueError: If x_names is not set.
        """
        if not self._x_names:
            raise ValueError("x_names not set")

    def check_y_name(self):
        """
        Check if the y_name attribute is set.

        Raises:
            ValueError: If y_name is not set.
        """
        if not self._y_name:
            raise ValueError("y_name not set")

    def setDataFrame(self, dataframe: T) -> None:
        """
        Sets the dataframe for the dataset.

        Parameters:
        dataframe (T): The dataframe to be set.

        Returns:
        None
        """
        self._dataframe = dataframe

    def getDataFrame(self) -> T:
        """
        Get the original dataframe.

        Returns:
            pd.DataFrame: The original dataframe.
        """
        return self._dataframe
    
    def setXNames(self, x_names: list[str]) -> None:
        """
        Set the list of column names to be used as input features.

        Args:
            x_names (list[str]): The list of column names.
        """
        self._x_names = x_names

    def setYName(self, y_name: str) -> None:
        """
        Set the column name to be used as the target variable.

        Args:
            y_name (str): The column name.
        """
        self._y_name = y_name

    def modify(self, func : Callable[[T], T]) -> None:
            """
            Modify the internal DataFrame

            Args:
                func (Callable[[pd.DataFrame], pd.DataFrame]): The function to be modified.
                    A callable that takes a pandas DataFrame as input and returns a modified DataFrame.

            Returns:
                None

            Raises:
                None
            """
            self._dataframe = func(self._dataframe)

    def view(self, func : Callable[[T], None]) -> None:
        """
        Applies the given function to the internal dataframe.

        Args:
            func (Callable[[pd.DataFrame], None]): The function to be applied to the dataframe.

        Returns:
            None
        """
        func(self._dataframe)
    
    @abstractmethod
    def getX(self) -> T:
        """
        Returns the input data (X) for the dataset.

        Returns:
            T: The input data (X) for the dataset.
        """
        pass

    @abstractmethod
    def getY(self) -> T:
        """
        Returns the Y value associated with the dataset.

        Returns:
            T: The Y value associated with the dataset.
        """
        pass

    @abstractmethod
    def getFilteredDataFrame(self) -> T:
        """
        Returns a filtered DataFrame based on certain criteria.

        Returns:
            T: The filtered DataFrame.
        """
        pass

    @abstractmethod
    def get_shape(self) -> tuple:
        """
        Returns the shape of the dataset.

        Returns:
            tuple: A tuple representing the shape of the dataset.
        """
        pass
    
    @abstractmethod
    def preview(self, n: int = 5) -> None:
        """
        Preview the first n rows of the dataset.

        Args:
            n (int): The number of rows to preview. Default is 5.
        """
        pass
