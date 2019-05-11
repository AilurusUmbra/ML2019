import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 4-fold  CV
X = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 2])
y = np.array([2.5, 2.5, 0.5, 0, -0.5, -1.5, -2.5, -4])
num_of_folds = 4
# Here, the parameter 'shuffle=True' provides the random permutation on the dataset.
kf = KFold(n_splits=num_of_folds, shuffle=True)

# regression result table
row = ['fold1', 'fold2', 'fold3', 'fold4']
col = ['Linear (deg=1)', 'Poly (deg=5)']
trainResult = pd.DataFrame(index=row, columns=col)
testResult = pd.DataFrame(index=row, columns=col)
trainResult.index.name = "Training MAE"
testResult.index.name = "Testing MAE"

# regression

colors = ['red', 'orange', 'teal', 'yellowgreen', 'blue']
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
i = 0
c = 0

# plot training points
plt.subplot(1,2,1)
plt.scatter(X, y, color='navy', s=30, marker='o', label="dataset")
plt.title('linear', fontsize=26)

plt.subplot(1,2,2)
plt.scatter(X, y, color='navy', s=30, marker='o', label="dataset")
plt.title('poly', fontsize=26)

# Iterate each fold
for train_index, test_index in kf.split(X):
    print('fold '+str(i+1), ": {TRAIN:", train_index, ", TEST:", test_index, "}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    xLimit = X_train.min()-2.5, X_train.max()+2.5
    yLimit = y_train.min()-2.5, y_train.max()+2.5

    # use linspace for plot regression line
    x_plot = np.linspace(xLimit[0], xLimit[1], 100)

    # create matrix versions of these arrays
    X_train = X_train[:, np.newaxis]
    X_test = X_test[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]


    # set limit of axis
    plt.xlim(xLimit)
    plt.ylim(yLimit)

    # Train & Predict
    for count, degree in enumerate([1, 5]):

        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        y_plot = model.predict(X_plot)  # predict regression line
        plt.subplot(1,2,count+1)
        plt.plot(x_plot, y_plot, color=colors[c], label="fold %d" % (i+1) )
        plt.legend(loc='lower left')
        # Record MAE
        trainResult.iloc[i-1, count] = mean_absolute_error(y_train, model.predict(X_train))
        testResult.iloc[i-1, count] = mean_absolute_error(y_test, model.predict(X_test))

    c = (c + 1) % 5
    i += 1
#plt.savefig('ML3-1.png')
plt.show()

# show MAE
print('-'*50)
print(trainResult.to_string())
print('-'*50)
print(testResult.to_string())
#display(trainResult, testResult)
