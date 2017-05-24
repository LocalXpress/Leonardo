import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import gaussian_process

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='linear_model')
parser.add_argument('-featuredim', type=int, default=20)
parser.add_argument('-inputfeatures', type=str, default='../Data/features_ALL.txt')
parser.add_argument('-labels', type=str, default='../Data/ratings.txt')
args = parser.parse_args()


features = np.loadtxt(args.inputfeatures, delimiter=',')
features_train = features[0:-5]
features_test = features[-5:]

pca = decomposition.PCA(n_components=args.featuredim)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

ratings = np.loadtxt(args.labels, delimiter=',')
ratings_train = ratings[0:-5]
ratings_test = ratings[-5:]

if args.model == 'linear_model':
	regr = linear_model.LinearRegression()
elif args.model == 'svm':
	regr = svm.SVR()
elif args.model == 'rf':
	regr = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
elif args.model == 'gpr':
	regr = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
else:
	raise NameError('Unknown machine learning model. Please us one of: rf, svm, linear_model, gpr')

regr.fit(features_train, ratings_train)
ratings_predict = regr.predict(features_test)


corr = np.corrcoef(ratings_predict, ratings_test)[0, 1]
print 'Correlation: ', corr

residue = np.mean((ratings_predict - ratings_test) ** 2)
print 'Residue: ', residue


print("Image||Ground-Truth||Prediction||Accuracy")
for i in range(0, 5):
	acc=(100-math.fabs(((ratings_predict[i]-ratings_test[i])/ratings_test[i])*100))
	print(str(i)+"       "+str("%.1f" % ratings_test[i])+"             "+str("%.1f" % ratings_predict[i])+"        "+str("%.1f" %acc))

truth, = plt.plot(ratings_test, 'r')
prediction, = plt.plot(ratings_predict, 'b')
plt.legend([truth, prediction], ["Ground Truth", "Prediction"])

plt.show()
