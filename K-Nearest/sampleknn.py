from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# load and prepare our dataset

iris = load_iris()
x = iris.data[:, : 10]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create the knn classifier
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')

# train the model
knn.fit(x_train, y_train)

# make predictions
pred = knn.predict(x_test)
print(pred)
print(accuracy_score(y_test, pred))
