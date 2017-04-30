import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import progressbar
from tabulate import tabulate
import pickle
from scipy.special import digamma, loggamma


class IBM1():

	def __init__(self):
		self.english = []  			# all sentences in the english text [[]]
		self.french = []  			# all sentences in the french text type [[]]
		self.V_e = set()  			# english vocabulary
		self.V_f = set()  			# french vocabulary
		self.V_e_indices = dict()  	# word -> index dictionary
		self.V_f_indices = dict()  	# word -> index dictionary
		self.V_e_words = dict()  	# index -> word dictionary
		self.V_f_words = dict()  	# index -> word dictionary
		self.V_e_size = int
		self.V_f_size = int
		self.t = None 				# translation probabilities stored in np array
		self.likelihoods = []		# to store log-likelihoods per epoch
		self.elbos = []				# to store ELBO per epoch		
		self.found_french_UNKs = 0 	# how many unknown french words were found in the test set
		self.found_english_UNKs = 0 # how many unknown english words were found in the test set
		self.null = True			# using NULL words
		self.null_generations = [] 	# total number of NULL alignments predicted in test set oer epoch

	def read_data(self, english_path, french_path, max_sents=np.inf, null=True, UNK=True, test_repr=False):
		"""
		Read the data in path and represent this as a list of lists
		in english and french.
		"""
		print('Reading data...')

		self.null = null

		print('Using NULL word: {}'.format(self.null))

		e = open(english_path, 'r')
		for k, line in enumerate(e):
			if k + 1 > max_sents:
				break
			sent = line.split()
			if self.null:
				# if we use NULL words we prepend all sentences with NULL
				sent = ['-NULL-'] + sent 
			self.english.append(sent)
			self.V_e.update(sent)  # add words to vocabulary
		if UNK:	
			self.fix_english_UNKs(10)
		self.V_e_size = len(self.V_e)
		e.close()


		f = open(french_path, 'r')
		for k, line in enumerate(f):
			if k + 1 > max_sents:
				break
			sent = line.split()
			self.french.append(sent)
			self.V_f.update(sent)  # add words to vocabulary	
		if UNK:	
			self.fix_french_UNKs(10)
		self.V_f_size = len(self.V_f)
		f.close()

		# make word <-> index dictionaries
		for index, f in enumerate(self.V_f):
			self.V_f_indices[f] = index
			self.V_f_words[index] = f

		# make word <-> index dictionaries
		for index, e in enumerate(self.V_e):
			self.V_e_indices[e] = index
			self.V_e_words[index] = e

		# initialize a translation matrix t
		self.initialize_t()

		if test_repr == True:
			print(
			'Finished. A total of {0} sentences, {1} French words, and {2} English words\n'.format(len(self.french),
																								   self.V_f_size,
																								   self.V_e_size))
			print ('Maximal French sentence length: {0}\nMaximal English sentence length: {1}'.format(self.m, self.l))
			print('Testing data representation:')
			print(self.english[100])
			print(self.french[100])
			print('\n')

		print('Finished reading data')

	def fix_english_UNKs(self, k):
		"""
		Replaces k English words that occur only once in the training set with -UNK-
		"""
		# flatten out list of lists into one long list
		word_counts = Counter((word for sentence in self.english for word in sentence))
		# get 10 of the words that occur once
		low = [word for word, count in word_counts.items() if count==1][0:k]
		# replace all words in low with -UNK-
		for i, sentence in enumerate(self.english):
			for j, word in enumerate(sentence):
				if self.english[i][j] in low:
					self.english[i][j] = '-UNK-'
		# remove all low words from vocabulary
		self.V_e = self.V_e - set(low)
		# but add -UNK-
		self.V_e.add('-UNK-')

	def fix_french_UNKs(self, k):
		"""
		Replaces k French words that occur only once in the training set with -UNK-
		"""
		# flatten out list of lists into one long list
		word_counts = Counter((word for sentence in self.french for word in sentence))
		# get 10 of the words that occur once
		low = [word for word, count in word_counts.items() if count==1][0:k]
		# replace all words in low with -UNK-
		for i, sentence in enumerate(self.french):
			for j, word in enumerate(sentence):
				if self.french[i][j] in low:
					self.french[i][j] = '-UNK-'
		# remove all low words from vocabulary
		self.V_f = self.V_f - set(low)
		# but add -UNK-
		self.V_f.add('-UNK-')


	def initialize_t(self):
		"""
		For each f and e initializes t(f|e) = 1 / |V_e|.
		"""
		print("Initializing t")
		self.t = 1. / self.V_e_size * np.ones((self.V_f_size, self.V_e_size))

	def update_t(self, c_ef, c_e):
		"""
		Updates t using the rule t(e|f) = c(e,f) / c(e)
		"""
		print("Updating t")
		self.t = np.multiply(c_ef, c_e ** -1)


	def epoch(self, log=False):
		"""
		Run one epoch of EM on self.english and self.french.
		"""

		c_ef = np.zeros(self.t.shape, dtype=np.float)
		c_e = np.zeros((1, self.V_e_size), dtype=np.float)

		bar = progressbar.ProgressBar(max_value=len(self.english))

		# Iterate over all sentence pairs.
		for k, (E, F) in enumerate(zip(self.english, self.french)):

			bar.update(k)

			F_indices = [self.V_f_indices[i] for i in F]
			E_indices = [self.V_e_indices[j] for j in E]

			for f in F_indices:

				normalizer = float(sum([self.t[f, w] for w in E_indices]))

				for e in E_indices:
					delta = self.t[f, e] / normalizer

					c_ef[f, e] += delta  # c_ef same shape as t
					c_e[0, e] += delta

			bar.update(k + 1)

		bar.finish()

		self.update_t(c_ef, c_e)

		likelihood = self.log_likelihood()
		self.likelihoods.append(likelihood)

		if log: print('Likelihood: {}'.format(likelihood))


	def update_t_VI(self, lmbda_fe, alpha):
		"""
		Updates t using the rule: 

			t(e|f) = exp(digamma(lmbda_fe[f, e]) - digamma(sum_f' lmbda_fe[f', e]))

		and since the f' are held in the 0th axis:

			sum_f' lmbda_fe[f', e] = np.sum(lmbda_fe, axis=0, keepdims=True))
		"""
		print("Updating t using VI")
		self.t = np.exp(digamma(lmbda_fe) - digamma(np.sum(lmbda_fe, axis=0, keepdims=True)))


	def epoch_VI(self, alpha=0.2, log=False, ELBO=False):
		"""
		Epoch but with Variational Inference instead of EM.
		"""

		lmbda_fe = alpha * np.ones(self.t.shape, dtype=np.float) # lmbda_{f|e}
		lmbda_e = alpha * np.ones((1, self.V_e_size), dtype=np.float) # sum_f' lmbda_{f'|e}

		bar = progressbar.ProgressBar(max_value=len(self.english))

		# Iterate over all sentence pairs.
		for k, (E, F) in enumerate(zip(self.english, self.french)):

			bar.update(k)

			F_indices = [self.V_f_indices[i] for i in F]
			E_indices = [self.V_e_indices[j] for j in E]

			for f in F_indices:

				for e in E_indices:
					# Add expected count for one (f,e) event:
					lmbda_fe[f, e] += self.t[f, e]  

			bar.update(k + 1)

		bar.finish()

		self.update_t_VI(lmbda_fe, alpha)

		if ELBO:
			elbo = self.elbo(lmbda_fe, alpha)
			self.elbos.append(elbo)

			if log: print('ELBO: {}'.format(elbo))

	def log_likelihood(self, add_constant=False):
		"""
		Computes log-likelihood of dataset under current 
		parameter-assignments self.t.
		Formula (7) in Schulz's tutorial:

			log p(f_1^m, a_1^m, m | e_0^l) \propto sum_{j=1}^m log p(f_j | e_{a_j})

		Where we've dropped the constant log p(a_j | m, l) = log(1 / (l+1)^m).
		This is added when add_constant=True.
		"""
		likelihood = 0
		for k, (E, F) in enumerate(zip(self.english, self.french)):
			alignment = self.align(F, E)
			l = 0
			F_indices = [self.V_f_indices[i] for i in F]
			E_indices = [self.V_e_indices[j] for j in E]
			for a, f in enumerate(F_indices):
				l += np.log(self.t[f, E_indices[alignment[a]]])
			likelihood += l
			
			if add_constant: 
				likelihood += -len(F) * np.log(len(E) + 1) # optional addition of log(1 / (l+1)^m)

		return likelihood

	def elbo(self, lmbda_fe, alpha):
		print('Computing the ELBO')
		likelihood = self.log_likelihood()

		KL = (np.sum(np.multiply(self.t, -1*lmbda_fe + alpha) + loggamma(lmbda_fe), axis=0, keepdims=True) - 
					self.V_f_size * loggamma(alpha) +
					loggamma(self.V_f_size * alpha) -
					loggamma(np.sum(lmbda_fe, axis=0, keepdims=True)))

		summed = np.sum(KL, axis=1)[0]
		if type(summed)==complex:
			summed = summed.real
		summed = summed.real

		return likelihood + summed

	def plot_likelihoods(self, path):
		"""
		Plot the likelihoods and save to path.
		"""
		plt.plot(range(len(self.likelihoods)), self.likelihoods)
		plt.savefig(path)
		plt.clf()

	def plot_elbos(self, path):
		"""
		Plot the likelihoods and save to path.
		"""
		plt.plot(range(len(self.elbos)), self.elbos)
		plt.savefig(path)
		plt.clf()

	def tabulate_t(self, english_words, k=3):
		for w in english_words:
			print('\n')
			translations = zip([self.V_f_words[i] for i in range(self.t.shape[0])], self.t[:, self.V_e_indices[w]])
			best = sorted(translations, key=lambda x: x[1], reverse=True)[0:k]

			print(tabulate([[w, best[i][0], best[i][1] * 100] for i in range(k)],
						   headers=['Source', 'Translation', 'Probability']))

	def save_vocabulary(self, path):
		"""
		Saves vocabularies to path (is called only by save_t).
		"""
		f = open(path + 'V_e_indices.pkl', 'wb')
		pickle.dump(self.V_e_indices, f)
		f.close()
		f = open(path + 'V_f_indices.pkl', 'wb')
		pickle.dump(self.V_f_indices, f)
		f.close()
		f = open(path + 'V_e_words.pkl', 'wb')
		pickle.dump(self.V_e_words, f)
		f.close()
		f = open(path + 'V_f_words.pkl', 'wb')
		pickle.dump(self.V_f_words, f)
		f.close()

	def load_vocabulary(self, path):
		"""
		Loads vocabularies from path (is called only by load_t).
		"""
		f = open(path + 'V_e_indices.pkl', 'rb')
		self.V_e_indices = pickle.load(f)
		f.close()
		f = open(path + 'V_f_indices.pkl', 'rb')
		self.V_f_indices = pickle.load(f)
		f.close()
		f = open(path + 'V_e_words.pkl', 'rb')
		self.V_e_words = pickle.load(f)
		f.close()
		f = open(path + 'V_f_words.pkl', 'rb')
		self.V_f_words = pickle.load(f)
		f.close()
		# set vocabulary size
		self.V_f_size = len(self.V_f)
		self.V_e_size = len(self.V_e)

	def save_t(self, path, log=False):
		"""
		Saving translation matrix t using pickle.
		
		Note: We save only the nonzero entries of t
		to save space.

		Use protocol=4 for objects greater than 4Gb(!)
		We also save the vocabulary since the indexing
		of the matrix t depends on it.
		"""
		print('Saving t')
		self.save_vocabulary(path)
		f = open(path + 'transition-probs.pkl', 'wb')
		b = np.nonzero(self.t)
		entries = [self.t[i, j] for i, j in zip(b[0], b[1])]
		nonzero = list(zip(b[0], b[1], entries))

		len_nonzero = len(list(nonzero))
		size_t = self.t.shape[0] * self.t.shape[1]
		if log: print('Fraction of nonzero elements in t: {0} / {1} = {2}'.format(len_nonzero, size_t, float(len_nonzero) / size_t))

		pickle.dump(nonzero, f, protocol=4)
		f.close()

	def load_t(self, path):
		"""
		Load the nonzero entries of a translation matrix t as saved by save_t, 
		and restore it to original form as a matrix and sets self.t with this matrix
		"""
		self.load_vocabulary(path)
		f = open(path + 'transition-probs.pkl', 'rb')
		nonzero = pickle.load(f)
		f.close()
		self.t = np.zeros((self.V_f_size, self.V_e_size), dtype=np.float)
		print('Loading t')
		bar = progressbar.ProgressBar(max_value=len(list(nonzero)))

		for k, (i, j, v) in enumerate(nonzero):
			bar.update(k)
			self.t[i, j] = v
			bar.update(k + 1)

		bar.finish()
		print('\nLoaded t from path\t{}\n'.format(path))

	def posterior(self, f, E_indices):
		"""
		P(a_j = i | f_j, e_0,...,e_l) = t(f_j | e_i) / sum_{i=0}^l t(f_j | e_i)
		"""
		return self.t[f, E_indices] / np.sum(self.t[f, E_indices])

	def align(self, F, E):
		"""
		F is French sentence with words (not indices)
		E is English sentence with words (not indices)
		"""
		F_indices = []
		for f in F:
			try:
				i = self.V_f_indices[f]
			except KeyError:
				i = self.V_f_indices["-UNK-"]  # handling unknown words
				self.found_french_UNKs += 1
			F_indices.append(i)
		
		E_indices = []
		for e in E:
			try:
				i = self.V_e_indices[e]
			except KeyError:
				i = self.V_e_indices["-UNK-"]  # handling unknown words
				self.found_english_UNKs += 1
			E_indices.append(i)

		alignment = []
		for f in F_indices:
			p = self.posterior(f, E_indices)
			a_f = np.argmax(p)
			alignment.append(a_f)
				
		return alignment

	def predict_alignment(self, french_testpath, english_testpath, outpath):
		self.found_french_UNKs = 0
		self.found_english_UNKs = 0
		f_testfile = open(french_testpath, 'r')
		e_testfile = open(english_testpath, 'r')
		f_sents = []
		e_sents = []
		for line in f_testfile:
			f_sents.append(line.split())
		for line in e_testfile:
			if self.null:
				e_sents.append(['-NULL-'] + line.split())
			else:
				e_sents.append(line.split())

		alignments = []

		for F, E in zip(f_sents, e_sents):
			alignment = self.align(F, E)
			alignments.append(alignment)
		f_testfile.close()
		e_testfile.close()

		outfile = open(outpath, 'w')
		nulls = 0
		for k, alignment in enumerate(alignments):
			for f, e in enumerate(alignment):
				if self.null:
					if e != 0:
						outfile.write('{0} {1} {2} {3}\n'.format(k + 1, e, f + 1, 'S'))
					else:
						nulls += 1
				else:
					outfile.write('{0} {1} {2} {3}\n'.format(k + 1, e + 1, f + 1, 'S'))
		outfile.close()
		print('French UNKs found: {}'.format(self.found_french_UNKs))
		print('English UNKs found: {}'.format(self.found_english_UNKs))
		self.null_generations.append(nulls)


