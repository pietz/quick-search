import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class SimpleRanker:
	def __init__(self, dataset, vocab_size=2000):
		self.text = dataset
		self.vectorizer = CountVectorizer(max_features=vocab_size, binary=True)
		self.data = self.vectorizer.fit_transform(dataset).toarray()
	def search(self, query, k=10, return_idx=False):
		q_vec = self.vectorizer.transform([query]).toarray()[0]
		self.sim = (q_vec * self.data).sum(axis=1)
		idx = self.sim.argsort()[::-1][:k]
		if return_idx:
			return idx
		return [self.text[i] for i in idx]
		
class CountRanker:
	def __init__(self, dataset, vocab_size=2000):
		self.text = dataset
		self.vectorizer = CountVectorizer(max_features=vocab_size)
		self.data = self.vectorizer.fit_transform(dataset).toarray()
	def search(self, query, k=10, return_idx=False):
		q_vec = self.vectorizer.transform([query]).toarray()[0]
		self.sim = (q_vec * self.data).sum(axis=1)
		idx = self.sim.argsort()[::-1][:k]
		if return_idx:
			return idx
		return [self.text[i] for i in idx]
		
class TFRanker:
	def __init__(self, dataset, vocab_size=2000):
		self.text = dataset
		self.vectorizer = CountVectorizer(max_features=vocab_size)
		self.data = self.vectorizer.fit_transform(dataset).toarray()
		self.data = self.data / self.data.sum(axis=1)[..., None]
	def search(self, query, k=10, return_idx=False):
		q_vec = self.vectorizer.transform([query]).toarray()[0]
		self.sim = (q_vec * self.data).sum(axis=1)
		idx = self.sim.argsort()[::-1][:k]
		if return_idx:
			return idx
		return [self.text[i] for i in idx]
		
class TFIDFRanker:
	def __init__(self, dataset, vocab_size=2000):
		self.text = dataset
		self.vectorizer = TfidfVectorizer(max_features=vocab_size)
		self.data = self.vectorizer.fit_transform(dataset).toarray()
	def search(self, query, k=10, return_idx=False):
		q_vec = self.vectorizer.transform([query]).toarray()[0]
		self.sim = (q_vec * self.data).sum(axis=1)
		idx = self.sim.argsort()[::-1][:k]
		if return_idx:
			return idx
		return [self.text[i] for i in idx]
		
class SemanticRanker:
	def __init__(self, dataset, vocab_size=2000):
		self.text = dataset
		self.vectorizer = TfidfVectorizer(max_features=vocab_size)
		tmp_data = self.vectorizer.fit_transform(dataset).toarray()
		di = {}
		for row in open('glove.6B.50d.10k.txt').readlines():
			sample = row.split(' ')
			word, values = sample[0], sample[1:]
			di[word] = np.array([float(x) for x in values])
		self.w2v = np.zeros((vocab_size, 50))
		words = self.vectorizer.get_feature_names()
		for i, word in enumerate(words):
			if word in di:
				self.w2v[i] = di[word]
		self.data = tmp_data @ self.w2v
	def search(self, query, k=10, return_idx=False):
		q_vec = self.vectorizer.transform([query]).toarray()
		q_vec = q_vec @ self.w2v
		q_vec = q_vec[0]
		# Calculate the Cosine Similarity
		q_dot = np.sqrt(np.dot(q_vec, q_vec))
		x_dot = np.sqrt((self.data**2).sum(axis=1))
		self.sim = np.dot(q_vec, self.data.T) / ((q_dot * x_dot) + 1)
		idx = self.sim.argsort()[::-1][:k]
		if return_idx:
			return idx
		return [self.text[i] for i in idx]
