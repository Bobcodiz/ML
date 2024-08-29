import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes


diabetes = load_diabetes()
print(diabetes)
diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(diabetes.head())