import numpy
import time
import math

from scipy import misc
import matplotlib.pyplot as plt

f = misc.face(gray=True)
plt.imshow(f, cmap='gray')
plt.show()
F = numpy.fft.fft2(f)
I2 = numpy.absolute(F)
I2 = I2 / 1000000
plt.imshow(I2, cmap='gray')
plt.show()
exit()

from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
#img.save('my.png')
img.show()

exit()

def myown_fft(x):
    x = x.T
    assert int(math.log(x.shape[0], 2)) == math.log(x.shape[0], 2)
    def fft_rec(x):
        X = numpy.zeros(shape=x.shape, dtype='complex64')
        if x.shape[0] == 2:
            X[0] = x[0] + x[1]
            X[1] = x[0] - x[1]
        else:
            N = x.shape[0]
            Em = fft_rec(x[0:N:2])
            Om = fft_rec(x[1:N:2])
            e = numpy.exp(-2 * numpy.pi * 1j * numpy.array(range(int(N/2))) / N)
            X[:int(N/2)] = Em + (e * Om.T).T
            X[int(N/2):] = Em - (e * Om.T).T
        return X
    return fft_rec(x).T

def myown_fft2(x):
    return myown_fft(myown_fft(x).T).T

if 0:
    I = numpy.random.uniform(size=(1024 * 4, 1024 * 4))
    a = time.clock()
    #numpy.fft.fft2(I)
    myown_fft2(I)
    print(a, time.clock())
    exit()

if 0:
    x = numpy.array([0, 1, 2, 3, 0, 1, 5, 8])
    X = numpy.fft.fft(x)
    print(X)
    X = myown_fft(x)
    print(X)

if 1:
    ### fft od 2d image
    x = numpy.array([[8, 2, 6, 1],
                     [3, 4, 7, 1]])
    X = numpy.fft.fft2(x)
    print(X)
    X = myown_fft2(x)
    print(X)
