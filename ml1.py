import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.datasets import fetch_california_housing

warnings.simplefilter("ignore")
sns.set_theme(style="darkgrid")

# First way to import
# housing = fetch_california_housing()
# df = pd.DataFrame(housing.data, columns=housing.feature_names)
# df["target"] = pd.Series(housing.target)
# print(df.head(3))

from sklearn.datasets import fetch_openml

housing = fetch_openml(name="house_prices", as_frame=True)
X, y = fetch_openml(name="house_prices", return_X_y=True)  # Data-target decomposition
X = X.drop(columns=['Id'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
