import numpy
import time

def myown_fft(x):
    x = x.T
    assert x.shape[0] in set([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024 * 2, 1024 * 4])
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
    p2 = set([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024 * 2, 1024 * 4])
    assert x.shape[0] in p2 and x.shape[1]
    X = myown_fft(x)
    Y = myown_fft(X.T)
    return Y

if 1:
    I = numpy.random.uniform(size=(1024 * 4, 1024 * 4))
    a = time.clock()
    #numpy.fft.fft2(I)
    myown_fft2(I)
    print(a, time.clock())
    exit()

if 1:
    x = numpy.array([0, 1, 2, 3, 0, 1, 5, 8])
    X = numpy.fft.fft(x)
    print(X)
    X = myown_fft(x)
    print(X)

if 0:
    ### fft od 2d image
    x = numpy.array([[8, 2, 6, 1],
                     [3, 4, 7, 1],
                     [6, 2, 8, 3],
                     [1, 1, 2, 2]])
    #x = numpy.array([[4, 5], [2, 1]])
    X = numpy.fft.fft2(x)
    print(X)
    X = myown_fft2(x)
    print(X)
