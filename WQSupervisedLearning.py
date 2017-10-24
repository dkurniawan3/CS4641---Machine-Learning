'''Implementation of various machine learning
techniques over the wine quality dataset'''

# pylint: disable=C0103

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from helper import plot_learning_curve
from helper import plot_confusion_matrix

#Ignore matplotlib warnings
warnings.filterwarnings("ignore", module="matplotlib")

df = pd.read_csv('winequality-white.csv', delimiter=';', quotechar='"')
df['quality'] = np.where(df['quality'] >= 7, 1, 0)
X = df.ix[:, df.columns != 'quality']
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=50)

plt.hist(df['quality'], range=(0, 1))
plt.xlabel('Quality')
plt.xticks([0, 1])
plt.ylabel('Count')
plt.title('Distribution of Wine Quality Classes')
plt.show()

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Neural Networks
nn = MLPClassifier(solver='lbfgs', alpha=.001, activation='tanh',
                   hidden_layer_sizes=(11, 5, 3, 2, 10),
                   learning_rate="constant", early_stopping=True, max_iter=200)
nn.fit(X_train, y_train)
nn_predictions = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print 'Wine Quality Neural Network Accuracy:', nn_accuracy
print '-----------------'

cnf_matrix = confusion_matrix(y_test, nn_predictions)
plt.figure()
class_names = sorted(y_test.unique())
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Wine Quality Neural Network Confusion Matrix')
plt.show()

#DecisionTreeClassifier
singleDT = DecisionTreeClassifier(max_depth=43, max_features=5)
singleDT.fit(X_train, y_train)
DT_predictions = singleDT.predict(X_test)
DT_accuracy = accuracy_score(y_test, DT_predictions)
print 'Wine Quality Decision Tree Accuracy:', DT_accuracy
print '-----------------'

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
plot_learning_curve(singleDT, 'Wine Quality Decision Tree Learning Curve', X, y,
                    ylim=(0.5, 1.01), cv=cv, n_jobs=4)
plt.show()

#Boosting
AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=40)
AdaBoost.fit(X_train, y_train)
boosted_accuracy = AdaBoost.score(X_test, y_test)
print 'Wine Quality AdaBoost Accuracy:', boosted_accuracy
print '-----------------'

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
plot_learning_curve(AdaBoost, 'Wine Quality AdaBoost Learning Curve', X, y,
                    ylim=(0.5, 1.01), cv=cv, n_jobs=4)
plt.show()

#SVM
svm = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print 'Wine Quality SVM MSE:', mean_squared_error(y_test, svm_predictions)
print 'Wine Quality SVM Accuracy:', svm_accuracy
print '-----------------'

cnf_matrix = confusion_matrix(y_test, svm_predictions)
plt.figure()
class_names = sorted(y_test.unique())
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Wine Quality SVM Confusion Matrix')
plt.show()

#kNN
Kvals = range(1, 10)
accs = [0]*len(Kvals)

for i, k in enumerate(Kvals):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    accs[i] = accuracy_score(y_test, knn_predictions)
    print 'Wine Quality KNN Accuracy: ' + str(accs[i])
    print '-----------------'

# Plot results
plt.figure()
plt.title('Wine Quality KNN: Accuracy vs. K')
plt.plot(Kvals, accs, '-', label='Accuracy')
plt.legend()
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
plot_learning_curve(knn, 'Wine Quality kNN Learning Curve', X, y,
                    ylim=(0.50, 1.01), cv=cv, n_jobs=4)
plt.show()
