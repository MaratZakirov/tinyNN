import cPickle, gzip, numpy

class Dense:
    def __init__(self, insz, outsz):
        self.insz  = insz
        self.outsz = outsz
        # weighs
        self.W = numpy.random.uniform(low=-0.1, high=0.1, size=(insz, outsz))
        # averaged gradient
        self.dW = None
        self.lr = 0.001
    def forward(self, X, _):
        self.X = X
        self.Y = numpy.dot(self.X, self.W)
        return self.Y
    def backward(self, dY):
        batch_size = self.X.shape[0]
        assert dY.shape == (batch_size, self.outsz)
        assert self.X.shape == (batch_size, self.insz)
        self.dW = numpy.dot(self.X.T, dY) / batch_size
        self.dY = numpy.dot(dY, self.W.T)
        return self.dY
    def update(self):
        self.W -= self.lr * self.dW

class Activation:
    def __init__(self):
        self.func = numpy.tanh
        self.dfunc = lambda : 1 - self.Y ** 2
    def forward(self, X, _):
        self.X = X
        self.Y = self.func(X)
        return self.Y
    def backward(self, dY):
        return self.dfunc() * dY
    def update(self):
        pass

class Loss:
    def __init__(self):
        self.func = lambda X, out: numpy.sum(0.5 * (X - out) ** 2) / X.shape[0]
        self.dfunc = lambda : self.X - self.out
    def forward(self, X, out):
        self.X = X
        self.out = out
        return self.func(X, out)
    def backward(self, _):
        return self.dfunc()
    def update(self):
        pass

class Sequential:
    def __init__(self, stages):
        self.stages = stages
    def forward(self, X, Y, predict=False):
        for stage in self.stages[0 : len(self.stages) - int(predict)]:
            #print(stage)
            X = stage.forward(X, Y)
        return X
    def backward(self):
        dY = 1
        for stage in reversed(self.stages):
            #print(stage)
            dY = stage.backward(dY)
    def update(self):
        for stage in self.stages:
            stage.update()

class Learner:
    def __init__(self, model):
        self.model = model
    def train(self, X, Y, batch_size, epoches, valid_split):
        assert len(X) == len(Y)
        bidx = int(len(X) * (1 - valid_split))
        X_train = X[ : bidx]
        Y_train = Y[ : bidx]
        X_valid = X[bidx : ]
        Y_valid = Y[bidx : ]
        I = numpy.array(range(len(X_train)))
        for epoch in range(epoches):
            # shuffle data
            numpy.random.shuffle(I)
            X_train = X_train[I]
            Y_train = Y_train[I]
            # train on minibatches
            L_train = 0
            for i in range(0, len(X_train) - batch_size, batch_size):
                X_mb = X_train[i : i + batch_size]
                Y_mb = Y_train[i : i + batch_size]
                L_train += model.forward(X_mb, Y_mb)
                model.backward()
                model.update()
            L_train /= (len(X_train) / batch_size)
            # validation
            L_valid = model.forward(X_valid, Y_valid)
            print('Epoch:', epoch, 'train:', L_train, 'valid:', L_valid)

# Try on mnist data
def LoadData(size=10000):
    f = open('C:\data\mnist.pkl','rb')
    _, _, test_set = cPickle.load(f)
    X_data, Y_data_num = test_set
    X_data = X_data[: size]
    Y_data_num = Y_data_num[: size]
    Y_data = numpy.zeros(shape=(len(Y_data_num), 10))
    for i in range(len(Y_data_num)):
        Y_data[i][Y_data_num[i]] = 1.0
    return X_data, Y_data

X, Y = LoadData()
X_test = X[0:10]
Y_test = Y[0:10]
X = X[10:]
Y = Y[10:]

model = Sequential([Dense(784, 50), Activation(), Dense(50, 10), Activation(), Loss()])
lr = Learner(model)
lr.train(X, Y, 32, 50, 0.1)

R_test = model.forward(X_test, Y_test, True)
assert R_test.shape == Y_test.shape
for i in range(len(Y_test)):
    print('L:', numpy.argmax(Y_test[i]), 'R:', numpy.argmax(R_test[i]))