import pandas as pd


wine_data = pd.read_csv('/Listas de Exercício/Lista_Monitor/q2'
                        '/wine.csv', header=None)
wine_data = wine_data.rename(columns=wine_data.iloc[0]).iloc[1:]