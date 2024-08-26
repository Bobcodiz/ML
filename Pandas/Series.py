import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''x = np.random.uniform(size=100)
print(x)
y = pd.Series(x)
plt.scatter(x, y)
plt.show()
print(y)

my_list = [1, 2, 3, 4, 5, 6, 7]
arr = np.array(my_list)
plt.scatter(arr, arr)
plt.title("Scatter plot")
plt.ylabel("y")
plt.xlabel("x")
plt.show()
print(arr)'''

df = pd.read_csv('airtravel.csv')
print(df)
print(df.loc[0])
print(df[[df.columns[1], df.columns[2]]])
x = df.sort_values(by= df.columns[1], ascending=True)
y = df.groupby('Month')[df.columns[2]].mean()
print(x, y)

plt.show()
