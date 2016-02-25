import sys
import numpy as np
RNG = 10
np.random.seed(RNG)
from scipy import linalg
# from sklearn import cross_decomposition
from sklearn.utils import resample
from sklearn.datasets import make_classification
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

from unbalanced_dataset import UnderSampler, OverSampler


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 ## Output Type 3 (Type3) or Type 42 (TrueType)
rcParams['font.sans-serif'] = 'Arial'
import seaborn as sns
sns.set_style("whitegrid")


sys.path.append('../maayanlab_utils')
from plots import COLORS10


# def down_sampling(X, y):
# 	'''
# 	down-sampling the major clsases to get a balanced ratio of both classes
# 	'''
# 	classes, counts = np.unique(y, return_counts=True)
# 	class_counts = dict(zip(classes, counts))
# 	diff = abs(counts[0] - counts[1])
# 	if counts[0] < counts[1]:
# 		minor_class = classes[0]
# 	elif counts[0] == counts[1]:
# 		minor_class = None
# 	else:
# 		minor_class = classes[1]

# 	if minor_class is not None:
# 		minor_mask = y == minor_class
# 		X_sampled, y_sampled = resample(X[~minor_mask], y[~minor_mask], n_samples=np.min(counts))
# 		X = np.vstack((X[minor_mask], X_sampled))
# 		y = np.concatenate((y[minor_mask], y_sampled))
# 	return X, y

# def up_sampling(X, y):
# 	'''
# 	up-sampling the minor classes
# 	''' 
# 	classes, counts = np.unique(y, return_counts=True)
# 	class_counts = dict(zip(classes, counts))
# 	diff = abs(counts[0] - counts[1])
# 	if counts[0] < counts[1]:
# 		minor_class = classes[0]
# 	elif counts[0] == counts[1]:
# 		minor_class = None
# 	else:
# 		minor_class = classes[1]
# 	if minor_class is not None:
# 		minor_mask = np.where(y == minor_class)[0]
# 		up_sample_mask = np.random.choice(minor_mask, diff, replace=True)
# 		X_sampled = X[up_sample_mask]
# 		y_sampled = y[up_sample_mask]
# 		X = np.vstack((X, X_sampled))
# 		y = np.concatenate((y, y_sampled))
# 	return X, y

