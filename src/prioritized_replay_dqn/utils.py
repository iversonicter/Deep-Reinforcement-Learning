import scipy

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.0002


def preprocess(img):

    rgb = scipy.misc.imresize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    output = gray.astype('float32') / 128 - 1
    return output
