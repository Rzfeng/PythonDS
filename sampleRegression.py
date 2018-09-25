from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
'''
SVR Vs. Random Forest
RF needs larger number of instances to work the randomization concept.
If you have a small amount of data compared to possible variations of instances, then SVM is better
'''

boston = load_boston()
boston.keys()
type(boston['data'])
#print(boston['DESCR'])


clf = RandomForestRegressor()

#Training our algorithm to predict "target" from "data", features vs labels
clf.fit(boston['data'], boston['target'])

#See how accurate our model is
score = clf.score(boston['data'], boston['target'])

#Make predictions
row = boston['data'][17]
row.shape
row.reshape(-1, 13)

p = clf.predict(row.reshape(-1, 13))
a = boston['target'][17]

print("accuracy (R^2): " + str(score))
print("predicted value: " + str(p))
print("actual value: " + str(a))
print("_______THESE WERE OVERFITTED_______")
#fit to train, score for accuracy, predict to predict.
#everything that ends with an _ are learned, i.e. "base_estimator_", "n_features_"


##################Train and Test Data##################

#To prevent Overfitting, split model into training data and test data
#Train model on training data, test data to evaluate model

from sklearn.model_selection import train_test_split
#this randomly splits train and test data. Might not be good in all cases
X_train, X_test, y_train, y_test = train_test_split(boston['data'], boston['target'], test_size = 0.3)

#X_train is features, train is labels (target)
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("R^2 value: " + str(score))

#Other methods to split training/test data, such as k-fold
#Splitting data to train/test randomly isn't always good, such as finding credit card fraud


##################Preprocess Data##################
''' Need to preprocess data before feeding it into our model
'''
import pandas as pd
from sklearn.svm import SVR
from sklearn import preprocessing

df = pd.DataFrame(boston['data'], columns = boston['feature_names'])
df.max(axis=0)

##Because NOX vs TAX has a huge difference, Random Forest Regression handles this, but
##you can preprocess it using SVR

clf = SVR()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

Xs = preprocessing.scale(boston['data'])
df2 = pd.DataFrame(Xs, columns = boston['feature_names'])
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, boston['target'], test_size = 0.3)

#
clf = SVR()
clf.fit(Xs_train, ys_train)
score = clf.score(Xs_test, ys_test)
print(score)

#Reduce dimensionality
#PCA: principle component analysis
#We're going to reduce our 13 features to 5

from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
pca.fit(boston['data'])

Xp = pca.transform(boston['data'])
XpShape = Xp.shape
print(XpShape)

clf = RandomForestRegressor()
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, boston['target'], test_size = 0.3)
clf.fit(Xp_train, yp_train)
print("_____After reducing demineions to _____")
print(clf.score(Xp_test, yp_test))


'''
1.Initial preprocessing in Pandas
2. load data into dataframe,
3. Join values with another data source, fill missing values, etc
4. Extract data as NumPy array using dataframe.values
5. Use scikit learn'''




################Data Pipelines################
'''
1. Scale Data
2. Reduce number of dimensions
3. Use SVR

'''


print("_____Using Data Pipelines_____")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components = 5)),
    ('svr', SVR()),
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(score)
print("______Data Pipeline Steps______")
print(pipe.steps)

pipe.get_params()


'''Storing and retrieving models (objects) '''
import pickle
with open('model.pickle', 'wb') as out:
    pickle.dump(pipe, out)
with open('model.pickle', 'rb') as fp:
    pipe1 = pickle.load(fp)

print(pipe1.steps)
