'''
Implementation of Forest ensembles with up, down sampling for each tree
'''
from joblib import Parallel, delayed
from imblearn.under_sampling import RandomUnderSampler as UnderSampler
from imblearn.over_sampling import RandomOverSampler as OverSampler

from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import check_random_state, as_float_array, _get_n_jobs

from utils import *


MAX_INT = np.iinfo(np.int32).max

class BootstrapSampler(object):
	"""A very simple BootstrapSampler having a fit_sample method"""
	def __init__(self, random_state=None):
		self.random_state = random_state
	
	def fit_sample(self, X, y):
		n_samples = X.shape[0]
		random_instance = check_random_state(self.random_state)
		sample_indices = random_instance.randint(0, n_samples, n_samples)
		return X[sample_indices], y[sample_indices]

def _partition_estimators(n_estimators, n_jobs):
	"""Private function used to partition estimators between jobs."""
	# Compute the number of jobs
	n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

	# Partition estimators between jobs
	n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=np.int)
	n_estimators_per_job[:n_estimators % n_jobs] += 1
	starts = np.cumsum(n_estimators_per_job)

	return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _parallel_build_trees(tree, forest, X, y):
	if forest.sampling is None:
		sampler = BootstrapSampler(random_state=tree.random_state)
	elif forest.sampling == 'up':
		sampler = OverSampler(random_state=tree.random_state)
	elif forest.sampling == 'down':
		sampler = UnderSampler(random_state=tree.random_state)

	X_sample, y_sample = sampler.fit_sample(X, y)
	tree.fit(X_sample, y_sample, check_input=False)
	return tree

def _parallel_helper(obj, methodname, *args, **kwargs):
	"""Private helper to workaround Python 2 pickle limitations"""
	return getattr(obj, methodname)(*args, **kwargs)


class ResampleForestClassifier(BaseEstimator, MetaEstimatorMixin):
	"""docstring for ResampleForestClassifier"""
	def __init__(self, base_estimator, 
		n_estimators=10, 
		sampling=None, 
		n_jobs=1,
		random_state=None,
		verbose=False,
		**sampler_kwargs
		):
		
		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.sampling = sampling
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.verbose = verbose

		self.estimators_ = []


	def fit(self, X, y):
		random_state = check_random_state(self.random_state)
		trees = []
		for i in range(self.n_estimators):
			tree = clone(self.base_estimator)
			tree.set_params(random_state=random_state.randint(MAX_INT))
			trees.append(tree)

		trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
			delayed(_parallel_build_trees)(
				t, self, X, y) for t in trees
				)
		# Collect trained trees
		self.estimators_.extend(trees)
		# 
		self.classes_ = np.unique(y)
		return self

	def predict_proba(self, X):
		# Assign chunk of trees to jobs
		n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

		# Parallel loop
		all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading')(
			delayed(_parallel_helper)(
				e, 'predict_proba', X) 
			for e in self.estimators_)

		# Reduce
		proba = all_proba[0]
		for j in range(1, len(all_proba)):
			proba += all_proba[j]

		proba /= len(self.estimators_)
		return proba

	def predict(self, X):
		proba = self.predict_proba(X)
		return self.classes_.take(np.argmax(proba, axis=1), axis=0)


