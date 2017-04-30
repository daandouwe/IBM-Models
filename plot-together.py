from lib.util import read_list
import matplotlib.pyplot as plt


path = 'prediction/validation/IBM1/VI/VALIDATION/'

savepath = path

names = ['random-init-1', 'random-init-2', 'random-init-3', 'uniform-init', 'pretrained-init']
labels = ['random-1', 'random-2', 'random-3', 'uniform', 'pretrained']
# labels = ['EM', 'VI']
# names = ['EM-10k', 'VI-10k']

handles = []
for i, name in enumerate(names):
	aers = read_list(path + name + '/AERs')
	ax = plt.plot(range(len(aers[0:14])), aers[0:14], label=labels[i])
	handles.extend(ax)
plt.legend(handles, labels)
plt.savefig(savepath + 'AERs-together.pdf')
plt.clf() 

handles = []
for i, name in enumerate(names):
	aers = read_list(path + name + '/likelihoods')
	ax = plt.plot(range(len(aers[0:14])), aers[0:14], label=labels[i])
	handles.extend(ax)
plt.legend(handles, labels)
plt.savefig(savepath + 'likelihoods.pdf')
