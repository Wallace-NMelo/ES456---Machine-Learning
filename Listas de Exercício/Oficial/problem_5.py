# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:29:14 2020

@author: arthur
"""

import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from mlp import MLP

data = load_iris(as_frame=True)
X = data.data
y = data.target
training_set = pd.concat([X, y], axis=1)

mlp = MLP(X.shape, ([4, 2, 3]))
mlp.fit(epochs=100, data=training_set, batch_size=5, l_r=2.0)