import pandas as pd
from pathlib import Path


# Path of current directory
current_path = Path.cwd()

wine_data = pd.read_csv(current_path.joinpath('wine.csv'), header=None)
wine_data = wine_data.rename(columns=wine_data.iloc[0]).iloc[1:]

