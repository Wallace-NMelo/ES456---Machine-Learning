import random
import pandas as pd
from pathlib import Path


# Path of current directory
current_path = Path.cwd()
"""
 # Creating the dataset

def random_generator():
    number = round(random.random())
    return number, number == 1
    
with open('/home/wallace_nascimento/2020_3/Machine Learning/Listas de Exercício/Lista_Monitor/q1'
          '/data_3xor.txt', "w") as f:
    for i in range(5000):
        number_1, number_2, number_3 = random_generator(), random_generator(), random_generator()
        #number_3 = number_1[1] & number_2[1] # and
        #number_3 = number_1[1] | number_2[1] # or
        #number_3 = number_1[1] ^ number_2[1]  # xor
        number_4 = (number_1[1] ^ number_2[1]) ^ number_3[1]
        #f.write("{0} {1} {2}\n".format(number_1[0], number_2[0], int(number_3 == True)))
        f.write("{0} {1} {2} {3}\n".format(number_1[0], number_2[0], number_3[0], int(number_4 == True)))
"""
data_and_random = pd.read_csv(current_path.joinpath('dataset/data_and.txt'), sep=" ", header=None, names=["A", "B", "C"])

data_xor_random = pd.read_csv(current_path.joinpath('dataset/data_xor.txt'), sep=" ", header=None, names=["A", "B", "C"])

data_or_random = pd.read_csv(current_path.joinpath('dataset/data_or.txt'), sep=" ", header=None, names=["A", "B", "C"])

data_3xor_random = pd.read_csv(current_path.joinpath('dataset/data_3xor.txt'), sep=" ", header=None, names=["A", "B", "C", "D"])