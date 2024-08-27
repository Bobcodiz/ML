""" A company wants to predict the price of a house based on its features. They have collected data on 20
houses, and each house is described by three features:

* `Area`: The area of the house in square feet
* `NumRooms`: The number of rooms in the house
* `Price`: The original selling price of the house
The company wants to use KNN to predict the price of a new house with an area of 1300 square feet, 5
rooms, and unknown original selling price. They want to find the k-nearest neighbors (k=3) and calculate
their average price."""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = np.array([[1200, 3, 200000],
                 [1500, 4, 250000],
                 [1000, 2, 180000],
                 [1300, 5, 210000],
                 [1100, 3, 190000],
                 [1400, 4, 220000],
                 [1600, 6, 270000],
                 [1200, 4, 200000],
                 [1500, 5, 250000],
                 [1000, 3, 180000],
                 [1300, 6, 230000],
                 [1100, 2, 190000],
                 [1400, 5, 220000],
                 [1600, 7, 280000],
                 [1200, 5, 200000],
                 [1500, 6, 260000],
                 [1000, 4, 180000],
                 [1300, 3, 190000],
                 [1100, 4, 210000],
                 [1400, 2, 220000]])

x = data[:, :2]
y = data[:, 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

knn = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto')

knn.fit(scaled_x_train, y_train)

y_pred = knn.predict(scaled_x_test)
print(y_pred)

