from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class DataPreprocessor:
    """
    A class for preprocessing data for classification tasks.

    Args:
        cluster (list): The relabeled cluster values.
        data (pandas.DataFrame): The input data.
        target_col_name (str): The name of the target column.

    Methods:
        X_and_y(): Extract the selected columns and target column from the data based on correlation and p-value.
        pipeline(X, y): Perform data preprocessing pipeline, including train-test split, undersampling, oversampling, and scaling.

    Usage:
        data_preprocessor = DataPreprocessor(cluster, data, target_col_name)
        X, y = data_preprocessor.X_and_y()
        X_train, X_test, y_train, y_test = data_preprocessor.pipeline(X, y)
    """

    def __init__(self, cluster, data, target_col_name):
        """
        Initialize the DataPreprocessor object.

        Args:
            cluster (list): The relabeled cluster values.
            data (pandas.DataFrame): The input data.
            target_col_name (str): The name of the target column.
        """
        self.cluster = cluster
        self.data = data
        self.target_col_name = target_col_name

    def X_and_y(self):
        """
        Extract the selected columns and target column from the data based on correlation and p-value.

        Returns:
            tuple: A tuple containing the selected columns and target column.
                   - selected_columns (pandas.DataFrame): The selected columns from the data.
                   - target_column (pandas.Series): The target column from the data.
        """
        # Step 1: Specify the column name for which you want to calculate the correlation and p-value
        # Specify the target column name
        cluster_col_name = f'{self.target_col_name} Relabeled'

        # Step 2: Calculate the correlation and p-value with other columns
        selected_columns = []

        for col_name in self.data.iloc[:, 5:-1].columns:
            if self.data[col_name].std() != 0:
                _, p_value = pearsonr(self.data[cluster_col_name], self.data[col_name])
                if p_value < 0.05:  # Add a condition to include only columns with p-value < 0.05
                    selected_columns.append(col_name)

        return self.data[selected_columns], self.data[cluster_col_name]

    def pipeline(self, X, y):
        """
        Perform data preprocessing pipeline, including train-test split, undersampling, oversampling, and scaling.

        Args:
            X (pandas.DataFrame): The input features.
            y (pandas.Series): The target variable.

        Returns:
            tuple: A tuple containing the preprocessed train-test split data.
                   - X_train (numpy.ndarray): The preprocessed training features.
                   - X_test (numpy.ndarray): The preprocessed testing features.
                   - y_train (numpy.ndarray): The preprocessed training target.
                   - y_test (numpy.ndarray): The preprocessed testing target.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        undersampling_ratio = {i: int(0.5 * len(y_train[y_train == i])) for i in range(1, len(self.cluster) + 1)}

        # Create an instance of RandomOverSampler
        oversampler = RandomOverSampler()

        # Create an instance of RandomUnderSampler
        undersampler = RandomUnderSampler(sampling_strategy=undersampling_ratio)

        # Perform undersampling
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

        # Perform oversampling on the undersampled data
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        return X_train, X_test, y_train, y_test
