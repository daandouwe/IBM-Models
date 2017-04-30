from lib.IBM1 import IBM1
from lib.IBM2 import IBM2
import numpy as np
from lib.aer_import import test
from lib.util import write_list

english_path = 'training/hansards.36.2.e'
french_path = 'training/hansards.36.2.f'

modelpath = '../../models/IBM2/pretrained-init/'
savepath = 'prediction/test/IBM2/pretrained-init/'

best_aer = modelpath + '6-'
best_likelihood = modelpath + '14-'


# Predicting for best AER model

ibm = IBM1()
alignment_path 	= savepath + 'prediction-6'
ibm.read_data(english_path, french_path, null=True,  UNK=True, max_sents=np.inf, test_repr=False)
ibm.load_t(best_aer)
ibm.predict_alignment('testing/test/test.f', 
					  'testing/test/test.e', 
					  alignment_path)
aer = test('testing/test/test.wa.nonullalign', 
				   alignment_path)
print('Total NULL alignments: {}'.format(ibm.null_generations[-1]))
write_list([aer], savepath + 'AER-6')


# Predicting for best likelihood model

ibm = IBM1()
alignment_path 	= savepath + 'prediction-15'
ibm.read_data(english_path, french_path, null=True,  UNK=True, max_sents=np.inf, test_repr=False)
ibm.load_t(best_likelihood)
ibm.predict_alignment('testing/test/test.f', 
					  'testing/test/test.e', 
					  alignment_path)
aer = test('testing/test/test.wa.nonullalign', 
				   alignment_path)
print('Total NULL alignments: {}'.format(ibm.null_generations[-1]))
write_list([aer], savepath + 'AER-15')


