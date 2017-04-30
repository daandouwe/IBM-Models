import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import progressbar
from tabulate import tabulate
import pickle
from math import floor
from numpy.random import dirichlet


class IBM2():

	def __init__(self):
		self.english = [] 			# all sentences in the english text [[]]
		self.french = [] 			# all sentences in the french text type [[]]
		self.V_e = set() 			# english vocabulary
		self.V_f = set() 			# french vocabulary
		self.V_e_indices = dict() 	# word -> index dictionary
		self.V_f_indices = dict() 	# word -> index dictionary
		self.V_e_words = dict() 	# index -> word dictionary
		self.V_f_words = dict() 	# index -> word dictionary
		self.V_e_size = int 		
		self.V_f_size = int
		self.t = np.zeros((0,0)) 	# translation probabilities stored in np array
		self.jump = np.zeros((0,0)) # jump probabilities stored in np array
		self.likelihoods = []		# to store likelihoods per epoch
		self.found_french_UNKs = 0 	# how many unknown french words were found in the test set
		self.found_english_UNKs = 0 # how many unknown english words were found in the test set
		self.max_jump = 100			# maximal jump for the jump probabilities
		self.null = False			# using NULL words
		self.null_generations = [] 	# per epoch stores the total number of NULL alignments predicted for test set

	def read_data(self, english_path, french_path, max_sents=np.inf, null=True, UNK=True, random_init=False, test_repr=False):
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
		if random_init:
			self.random_initialize_t()
		else:
			self.initialize_t()
		# initialize a jump matrix jump
		self.initialize_jump()

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
		word_counts = Counter((word for sentence in self.english for word in sentence))
		# get 10 of the words that occur once
		low = [word for word, count in word_counts.items() if count==1][0:k]
		self.low_dict = {word: '-UNK-' for word in low}
		# replace all words in low with -UNK-
		for i, sentence in enumerate(self.english):
			for j, word in enumerate(sentence):
				if self.english[i][j] in low:
					self.english[i][j] = '-UNK-'
		# remove all low words from vocabulary
		self.V_e = self.V_e - set(low)
		self.V_e.add('-UNK-')

	def fix_french_UNKs(self, k):
		"""
		Replaces k French words that occur only once in the training set with -UNK-
		"""
		word_counts = Counter((word for sentence in self.french for word in sentence))
		# get 10 of the words that occur once
		low = [word for word, count in word_counts.items() if count==1][0:k]
		self.low_dict = {word: '-UNK-' for word in low}
		# replace all words in low with -UNK-
		for i, sentence in enumerate(self.french):
			for j, word in enumerate(sentence):
				if self.french[i][j] in low:
					self.french[i][j] = '-UNK-'
		# remove all low words from vocabulary
		self.V_f = self.V_f - set(low)
		self.V_f.add('-UNK-')


	def initialize_t(self):
		"""
		For each f and e initializes t(f|e) = 1 / |V_e|.
		"""
		print("Initializing t uniformly")
		self.t = 1. / self.V_e_size * np.ones((self.V_f_size, self.V_e_size))

	def random_initialize_t(self):
		"""
		Initialize the translations by drawing a Categorical distribution
		(t(f_1|e),...,t(f_{V_f_size}|e)) for each e from a Dirichlet distribution:
			
			(t(f_1|e),...,t(f_{V_f_size}|e)) ~ Dir(0.1,...,0.1).

		"""
		print("Initializing t randomly")
		self.t = dirichlet((0.1,) * self.V_f_size, size=self.V_e_size).T

	def update_t(self, c_ef, c_e):
		"""
		Updates t using the rule t(e|f) = c(e,f) / c(e)
		"""
		print("Updating t")
		self.t = np.multiply(c_ef, c_e ** -1)

	def initialize_jump(self):
		"""
		Initializes an array [0, 2 * max_jump] that we interpret as
		the array [-max_jump, max_jump]
		Initialization with uniform probabilities.
		"""
		print("Initializing jump")
		self.jump = 1. / (2 * self.max_jump) * np.ones((1, 2 * self.max_jump), dtype = np.float)

	def update_jump(self, c_jump):
		"""
		Normalizing the c_jump vector.
		Return c_jump / sum(c_jump)
		"""
		print("Updating jump")
		self.jump = 1./float(np.sum(c_jump)) * c_jump

	def jump_func(self, i, j, l, m):
		"""
		Conventions:
		We align french word j to english word i. 
		i = 0,...,l
		j = 1,...,m
		That is: a_j = i
		
		Returns a jump value in the range [0, 2*max_jump].
		We shifted[-max_jump, max_jump] to [0, 2*max_jump] to
		make indices compatible with an array.
		"""
		jump = int(i - floor(j * l / m)) + self.max_jump 
		if jump >= 2 * self.max_jump:
			return self.max_jump - 1
		if jump < 0:
			return 0
		else:
			return jump

	def epoch(self, log=False):
		"""
		Run one epoch of EM on self.english and self.french.
		"""
		c_ef = np.zeros(self.t.shape, dtype=np.float)
		c_e = np.zeros((1, self.V_e_size), dtype=np.float)

		c_jump = np.zeros((1, 2 * self.max_jump), dtype=np.float)

		bar = progressbar.ProgressBar(max_value = len(self.english))
		
		# Iterate over all sentence pairs.
		for k, (E, F) in enumerate(zip(self.english, self.french)):

			bar.update(k)
			
			m = len(F)

			F_indices = [self.V_f_indices[i] for i in F]
			E_indices =  [self.V_e_indices[j] for j in E]

			# TODO: change i, j convention: j ranges over French words; i over English words (see jump_function)
			for i, f in enumerate(F_indices):

				l = len(E)

				# normalizer = float(sum([self.t[f,w] * self.jump[self.jump_func(j, i, l, m)] for j, w in enumerate(E_indices)]))
				normalizer = 0.0
				for j, w in enumerate(E_indices):

					normalizer += self.t[f,w] * self.jump[0, self.jump_func(j, i, l, m)]

				for j, e in enumerate(E_indices):
					
					index = self.jump_func(j, i, l, m)

					if normalizer == 0:
						print('help!')
						delta = 0
					else:
						delta = self.t[f,e] * self.jump[0, index] / normalizer

					c_ef[f,e] += delta # c_ef same shape as t
					c_e[0,e] += delta
				
					c_jump[0, index] += delta

			bar.update(k+1)

		bar.finish()

		self.update_t(c_ef, c_e)
		self.update_jump(c_jump)

		likelihood = self.log_likelihood()
		self.likelihoods.append(likelihood)

		if log: print('Likelihood: {}'.format(likelihood))


	def log_likelihood(self):
		"""
		Computes log-likelihood of dataset under current  parameter-assignments self.t.
		Formula (14) in Schulz's tutorial:

			log p(f_1^m, a_1^m, m | e_0^l) \propto sum_{j=1}^m log p(i | j, m, l) + log p(f_j | e_{a_j})

		Note: we use a jump-function, hence p(i | j, m, l) = jump(i, j, l, m)
		"""
		likelihood = 0
		for k, (E, F) in enumerate(zip(self.english, self.french)):
			
			alignment = self.align(F, E)
			F_indices = [self.V_f_indices[i] for i in F]
			E_indices =  [self.V_e_indices[k] for k in E]
			l = len(E)
			m = len(F)
			
			lh = 0
			for j, f in enumerate(F_indices):
				lh += np.log(self.t[f, E_indices[alignment[j]]]) + np.log(self.jump[0, self.jump_func(alignment[j], j, l, m)])
			likelihood += lh

		return likelihood

	def plot_likelihoods(self, path):
		"""
		Plot the likelihoods and save to path.
		"""
		plt.plot(range(len(self.likelihoods)), self.likelihoods)
		plt.savefig(path)
		plt.clf()

	def tabulate_t(self, english_words, k=3):
		for w in english_words:
			print('\n')
			translations = zip([self.V_f_words[i] for i in range(self.t.shape[0])], self.t[:, self.V_e_indices[w]])
			best = sorted(translations, key=lambda x: x[1], reverse=True)[0:k]

			print(tabulate([[w, best[i][0], best[i][1] * 100] for i in range(k)], headers=['Source', 'Translation', 'Probability']))

	def save_vocabulary(self, path):
		"""
		Saves vocabularies to path (is called only by save_t).
		"""
		f = open(path+'V_e_indices.pkl', 'wb')
		pickle.dump(self.V_e_indices, f)
		f.close()
		f = open(path+'V_f_indices.pkl', 'wb')
		pickle.dump(self.V_f_indices, f)
		f.close()
		f = open(path+'V_e_words.pkl', 'wb')
		pickle.dump(self.V_e_words, f)
		f.close()
		f = open(path+'V_f_words.pkl', 'wb')
		pickle.dump(self.V_f_words, f)
		f.close()

	def load_vocabulary(self, path):
		"""
		Loads vocabularies from path (is called only by load_t).
		"""
		f = open(path+'V_e_indices.pkl', 'rb')
		self.V_e_indices = pickle.load(f)
		f.close()
		f = open(path+'V_f_indices.pkl', 'rb')
		self.V_f_indices = pickle.load(f)
		f.close()
		f = open(path+'V_e_words.pkl', 'rb')
		self.V_e_words = pickle.load(f)
		f.close()
		f = open(path+'V_f_words.pkl', 'rb')
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
		f = open(path+'transition-probs.pkl', 'wb')
		b = np.nonzero(self.t)
		entries = [self.t[i,j] for i,j in zip(b[0], b[1])]
		nonzero = list(zip(b[0], b[1], entries))
		
		len_nonzero = len(list(nonzero))
		size_t = self.t.shape[0] * self.t.shape[1]
		if log: print('Fraction of nonzero elements in t: {0} / {1} = {2}'.format(len_nonzero, size_t, float(len_nonzero)/ size_t))
		
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
		self.t = np.zeros((self.V_f_size, self.V_e_size), dtype = np.float)
		print('Loading t')
		bar = progressbar.ProgressBar(max_value=len(list(nonzero)))

		for k, (i,j,v) in enumerate(nonzero):
			bar.update(k)
			self.t[i,j] = v
			bar.update(k+1)

		bar.finish()

		print('Loaded t from path {}'.format(path + 'transition-probs.pkl'))

	def save_jump(self, path):
		"""
		Save the jump-probabilites matrix self.jump to path.
		"""
		f = open(path + 'jump-probs.pkl', 'wb')
		pickle.dump(self.jump, f)
		f.close()

	def load_jump(self, path):
		"""
		Load the jump-probabilites matrix self.jump from path
		and set max_jump to half the length of jump.
		"""
		f = open(path + 'jump-probs.pkl', 'rb')
		self.jump = pickle.load(f)
		self.max_jump = int(len(self.jump[0]) / 2)
		f.close()

	def posterior(self, f, j, E, F):
		"""
		Compute the posterior distribution of alignment.
		Formula given in Schulz page 8:

			P(a_j = i | f_j, e_i) = t(f_j | e_i) / sum_{i=0}^l t(f_j | e_i)
		
		"""
		normalize = 0.0
		m = len(F)
		l = len(E)
		p = []
		for i, e in enumerate(E):
			index = self.jump_func(i, j, l, m)
			prod = self.jump[0,index] * self.t[f,e]
			p.append(prod)
			normalize += prod
		return 1. / normalize * np.array(p)

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
		for j, f in enumerate(F_indices):
			p = self.posterior(f, j, E_indices, F_indices)
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

		self.null_generations.append(nulls)