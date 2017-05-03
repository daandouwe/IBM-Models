from lib.IBM1 import IBM1
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer
from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np


def main():

	ibm = IBM1()

	english_path = 'training/hansards.36.2.e'
	french_path = 'training/hansards.36.2.f'

	ibm.read_data(english_path, french_path, null=True, UNK=True, max_sents=1000, test_repr=False)
 
	Save = False

	T = 5
	aers = []
	for step in range(T):
		
		print('Iteration {}'.format(step+1))

		# setting saving paths
		save_path 		= 'prediction/validation/IBM1/VI/'
		model_path 		= '../../models/IBM1/VI/{0}-'.format(step+1)
		alignment_path 	= save_path + 'prediction-{0}'.format(step+1)

		ibm.epoch_VI(alpha=1, log=True, ELBO=False)

		ibm.predict_alignment('validation/dev.f', 
							  'validation/dev.e', 
							  alignment_path)
		
		aer = test('validation/dev.wa.nonullalign', 
				   alignment_path)
		
		aers.append(aer)
		print('AER: {}'.format(aer))

		print('Total NULL alignments: {}'.format(ibm.null_generations[-1]))

		# draw weighted alignments for sentence 21 (not working properly)
		# draw_weighted_alignment(ibm, alignment_path,
		# 							 '../validation/dev.f', 
		# 							 '../validation/dev.e', 
		# 							 '../prediction/validation/sentence-draws/IBM1-sentence-21-iter-{}'.format(step+1), 
		# 							 sentence=21)
		if Save:
			# save translation probabilities
			ibm.save_t(model_path)

	if Save:
		# save elbos
		write_list(ibm.elbos, save_path + 'ELBOs')
		# plot elbos
		ibm.plot_elbos(save_path + 'ELBOs.pdf')
		# save aers
		write_list(aers, save_path + 'AERs')
		# plot aers
		plot_aer(aers, save_path)
		# save total NULL alignments
		write_list(ibm.null_generations, save_path + 'NULL-generations')

	ibm.tabulate_t(english_words=['the', 'and', 'me', 'is', 'where', 'of', 'or', '-NULL-'], k=4)

if __name__ == "__main__":
	main()

