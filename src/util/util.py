import matplotlib as plt

def show_images(images, n=3):
    fig = plt.figure()

    for i in range(n):
        for j in range(n):
            subplot = fig.add_subplot(n, n, n*i+j+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            #subplot.set_title('%d' % (name))
            subplot.imshow(images[n*i+j].reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()