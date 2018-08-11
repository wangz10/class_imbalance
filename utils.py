import sys
import numpy as np
RNG = 10
np.random.seed(RNG)
import pandas as pd

from sklearn.utils import resample
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler as UnderSampler
from imblearn.over_sampling import RandomOverSampler as OverSampler


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 ## Output Type 3 (Type3) or Type 42 (TrueType)
rcParams['font.sans-serif'] = 'Arial'
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context('talk')


COLORS10 = [
'#1f77b4',
'#ff7f0e',
'#2ca02c',
'#d62728',
'#9467bd',
'#8c564b',
'#e377c2',
'#7f7f7f',
'#bcbd22',
'#17becf',
]

## Data loaders
def load_synthetic_data():
	X, y = make_classification(n_classes=2, class_sep=2, weights=[0.95, 0.05],
		n_informative=2, n_redundant=5, flip_y=0.05,
		n_features=50, n_clusters_per_class=1,
		n_samples=1000, random_state=RNG)	
	return X, y

def load_titanic():
	titanic = pd.read_csv('datasets/Titanic/train.csv')
	## Process features
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	for col in ['Sex', 'Cabin', 'Embarked']:
		titanic[col] = le.fit_transform(titanic[col])

	## Split df into X and y and convert to numpy.array
	X = titanic.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1).fillna(-1).values
	y = titanic['Survived'].values
	print X.shape, y.shape
	return X, y

def load_wpbc():
	bc = pd.read_csv('datasets/wpbc.data', 
		names=['ID', 'outcome'] + ['Attr%s'%i for i in range(33)])
	bc = bc.replace('?', -1)
	## Split df into X and y and convert to numpy.array
	X = bc.drop(['ID', 'outcome'], axis=1)
	y = bc['outcome'].map({'N':0, 'R':1})

	X = X.values
	y = y.values
	return X, y

def load_proteomics():
	proteome = pd.read_csv('datasets/Harmonizome/gene_attribute_matrix_cleaned.txt.gz', 
		sep='\t', compression='gzip', skiprows=2)
	## Load HGNC gene family 
	gene_family = pd.read_csv('datasets/Harmonizome/HGNC_gene_family.txt',sep='\t')
	# Left join with proteome data
	gene_family = gene_family[['Approved Symbol', 'Gene family description']]
	gene_family.set_index('Approved Symbol', inplace=True)

	proteome = proteome.drop(['UniprotAcc', 'GeneID/Brenda Tissue Ontology BTO:'], axis=1)
	proteome.set_index('GeneSym', inplace=True)

	proteome = proteome.merge(gene_family, left_index=True, right_index=True, how='inner')

	# Split X and y
	X = proteome.drop(['Gene family description'], axis=1)
	X = X.values
	y = proteome['Gene family description']
	y = map(lambda x: 'kinase' in x.lower(), y)
	y = np.array(y, dtype=np.int64)
	return X, y

def pca_plot(X, y):
	pca = PCA(n_components = 2)
	X_pc = pca.fit_transform(X)
	
	fig, ax = plt.subplots()
	mask = y==0
	ax.scatter(X_pc[mask, 0], X_pc[mask, 1], color=COLORS10[0], label='Class 0', alpha=0.5 ,s=20)
	ax.scatter(X_pc[~mask, 0], X_pc[~mask, 1], color=COLORS10[1], label='Class 1', alpha=0.5 ,s=20)
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.legend(loc='best')
	return fig


