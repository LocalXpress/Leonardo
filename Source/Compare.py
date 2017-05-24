import math
import argparse
import numpy as np
import matplotlib.pyplot as plt;plt.rcdefaults()
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
	regr_1 = linear_model.LinearRegression()
	regr_2 = svm.SVR()
	regr_3 = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
	regr_4 = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)


regr_1.fit(features_train, ratings_train)
ratings_predict_1 = regr_1.predict(features_test)

corr_1 = math.fabs(np.corrcoef(ratings_predict_1, ratings_test)[0, 1])
print 'Correlation:', corr_1

residue_1 = math.fabs(np.mean((ratings_predict_1 - ratings_test) ** 2))
print 'Residue:', residue_1

regr_2.fit(features_train, ratings_train)
ratings_predict_2 = regr_2.predict(features_test)

corr_2 = math.fabs(np.corrcoef(ratings_predict_2, ratings_test)[0, 1])
print 'Correlation:', corr_2

residue_2 = math.fabs(np.mean((ratings_predict_2 - ratings_test) ** 2))
print 'Residue:', residue_2

regr_3.fit(features_train, ratings_train)
ratings_predict_3 = regr_3.predict(features_test)

corr_3 = math.fabs(np.corrcoef(ratings_predict_3, ratings_test)[0, 1])
print 'Correlation:', corr_3

residue_3 = math.fabs(np.mean((ratings_predict_3 - ratings_test) ** 2))
print 'Residue:', residue_3

regr_4.fit(features_train, ratings_train)
ratings_predict_4 = regr_4.predict(features_test)

corr_4 = math.fabs(np.corrcoef(ratings_predict_4, ratings_test)[0, 1])
print 'Correlation:', corr_4

residue_4 = math.fabs(np.mean((ratings_predict_4 - ratings_test) ** 2))
print 'Residue:', residue_4

corr=np.array([corr_1,corr_2,corr_3,corr_4])
residue=np.array([residue_1,residue_2,residue_3,residue_4])


#import matplotlib.pyplot as plt; plt.rcdefaults()
#import numpy as np
#import matplotlib.pyplot as plt
 
objects = ('LM', 'SVM', 'RF', 'GPR')
y_pos = np.arange(len(objects))
 
plt.barh(y_pos, corr, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Pearson Coorelation')
plt.title('Comparision of different Machine Learning Algorithm')
 
plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as p
 
objects = ('LM', 'SVM', 'RF', 'GPR')
y_pos = np.arange(len(objects))
 
p.barh(y_pos, corr, align='center', alpha=0.5)
p.yticks(y_pos, objects)
p.xlabel('Residual')
p.title('Comparision of different Machine Learning Algorithm')
p.show()