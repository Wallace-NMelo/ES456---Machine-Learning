import pandas as pd
from pathlib import Path


# Path of current directory
current_path = Path.cwd()


# Iris Dataset
iris_data = pd.read_csv(current_path.joinpath('dataset/iris.data'),
                   sep=",", header=None, names=["sepal length", "sepal width", "petal length", "petal width", "Class"])
# Wine Dataset
wine_columns = ['Type', 'Alcohol', 'Malic', 'Ash',
                      'Alcalinity', 'Magnesium', 'Phenols',
                      'Flavanoids', 'Nonflavanoids',
                      'Proanthocyanins', 'Color', 'Hue',
                      'Dilution', 'Proline']

wine_data = pd.read_csv(current_path.joinpath('dataset/wine.data'),
                   sep=",", header=None, names=wine_columns)


# Pulse Dataset
pulse_data = pd.read_csv(current_path.joinpath('dataset/pulsos.csv'),
                         sep=',', header=None)
