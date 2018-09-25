from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
boston.keys()
type(boston['data'])
print(boston['DESCR'])


clf = RandomForestRegressor()

#Training our algorithm to predict "target" from "data", features vs labels
clf.fit(boston['data'], boston['target'])


#See how accurate our model is
clf.score(boston['data'], boston['target'])
