import numpy

def myown_fft(x):
    assert x.shape[0] in set([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
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
            X[:int(N/2)] = Em + e * Om
            X[int(N/2):] = Em - e * Om
        return X
    return fft_rec(x)

x = numpy.array([0, 1, 2, 3, 0, 1, 5, 8])
X = numpy.fft.fft(x)
print(X)
X = myown_fft(x)
print(X)
