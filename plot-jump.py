from lib.IBM2 import IBM2
import matplotlib.pyplot as plt

# plot a sequence of jump-distributions, per iteration.

save_path = 'prediction/validation/IBM2/uniform-init/'

ibm = IBM2()
handles = []

# Uncomment if you want the initial uniform distribution as a plot as well:
# ibm.load_jump('../../models/1-') # to get max-jump value
# # initialize uniformly
# ibm.initialize_jump()
# xs = list(map(lambda x : x - ibm.max_jump, list(range(len(ibm.jump[0])))))
# ax = plt.plot(xs, ibm.jump[0], label=0)
# handles.extend(ax)

for step in range(5):
	ibm.load_jump('../../models/IBM2/uniform-init/{0}-'.format(step+1))
	xs = list(map(lambda x : x - ibm.max_jump, list(range(len(ibm.jump[0])))))
	ax = plt.plot(xs, ibm.jump[0], label=step+1, linewidth=1)
	handles.extend(ax)
plt.legend(handles=handles)
plt.savefig(save_path + 'epoch-jump-plot.pdf')
plt.clf()