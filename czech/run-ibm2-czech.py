from lib.IBM2 import IBM2
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer, plot_jump
from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np


def main():

	ibm = IBM2()

	english_path = 'czech/news-commentary-v9.cs-en.en'
	czech_path = 'czech/news-commentary-v9.cs-en.cs'


	ibm.read_data(english_path, czech_path, null=True,  UNK=True, max_sents=20000, random_init=False, test_repr=False)
	
	# optional loading of 10 epochs pretrained translation matrix:
	# ibm.load_t('../../models/IBM1/NULL/5-') (BAD PATH!)

	Save = True

	T = 5
	aers = []
	for step in range(T):
		
		print('Iteration {}'.format(step+1))

		# setting saving paths
		save_path 		= 'czech/'
		model_path 		= '../../models/czech/{0}-'.format(step+1)
		alignment_path 	= save_path + 'prediction-{0}'.format(step+1)
		
		ibm.epoch(log=True)

		# ibm.predict_alignment('validation/dev.f', 
		# 					  'validation/dev.e', 
		# 					  alignment_path)
		
		# aer = test('validation/dev.wa.nonullalign', 
		# # 		   alignment_path)
		
		# aers.append(aer)
		# print('AER: {}'.format(aer))

		# draw weighted alignments for sentence 21 (not working properly)
		# draw_weighted_alignment(ibm, alignment_path,
		# 							 '../validation/dev.f', 
		# 							 '../validation/dev.e', 
		# 							 '../prediction/validation/sentence-draws/IBM1-sentence-21-iter-{}'.format(step+1), 
		# 							 sentence=21)
		if Save:		
			# save translation probabilities
			ibm.save_t(model_path)
			# save jump probabilities
			ibm.save_jump(model_path)

	# if Save:
		# save likelihoods
		# write_list(ibm.likelihoods, save_path + 'likelihoods')
		# plot likelihoods
		# ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')
		# save aers
		# write_list(aers, save_path + 'AERs')
		# plot aers
		# plot_aer(aers, save_path)
		# plot jump distribution
		# plot_jump(ibm.jump, ibm.max_jump, save_path)

	# ibm.tabulate_t(english_words=['the', 'and', 'me', 'is', 'where', 'of', 'or', '-NULL-'], k=4)


if __name__ == "__main__":
	main()
