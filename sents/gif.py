import imageio
images = []

filenames = ['sents-{}.png'.format(i) for i in range(5)]
# filenames = ['loaded/IBM2/sents-{}.png'.format(i) for i in range(11)]

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('sents0-5.gif', images, duration=0.7)
# imageio.mimsave('loaded/IBM2/sents-movie.gif', images, duration=0.7)