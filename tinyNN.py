import numpy
import functools
import operator

class Dense:
    def __init__(self, insz, outsz):
        self.insz  = insz
        self.outsz = outsz
        # weighs
        self.W = numpy.random.uniform(low=-0.5, high=0.5, size=(insz, outsz))
        # averaged gradient
        self.dW = numpy.zeros(shape=self.W.shape)
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

class Optimizer(object):
    def __init__(self, model):
        self.model = model
        self.load()

    # vectorize from model
    def load(self):
        self.W  = []
        self.dW = []
        for stage in self.model.stages:
            if hasattr(stage, 'W'):
                assert stage.W.shape == stage.dW.shape
                stage_sz = functools.reduce(operator.mul, list(stage.W.shape))
                self.W.append(numpy.reshape(stage.W, newshape=stage_sz))
                self.dW.append(numpy.reshape(stage.dW, newshape=stage_sz))
        self.W = numpy.concatenate(self.W)
        self.dW = numpy.concatenate(self.dW)

    # unvectorize into model
    def store(self):
        b = 0
        for stage in self.model.stages:
            if hasattr(stage, 'W'):
                assert stage.W.shape == stage.dW.shape
                a = b
                b += functools.reduce(operator.mul, list(stage.W.shape))
                stage.W = numpy.reshape(self.W[a : b], newshape=stage.W.shape)

# Very strong and resource consuming optimizer
class SGD(Optimizer):
    def __init__(self, model, lr=0.001, m=0.95):
        super(SGD, self).__init__(model)
        self.V = numpy.zeros(shape=self.W.shape)
        self.lr = lr
        self.m = m

    def step(self):
        self.load()
        self.V = self.m * self.V - self.lr * self.dW
        self.W += self.V
        self.store()

# Complex optimizatiion method according to Nocedal book
class LBFGS(Optimizer):
    def __init__(self, model, alpha=0.3, hist_size=10):
        super(LBFGS, self).__init__(model)
        self.hist_size = hist_size
        self.s = numpy.zeros(shape=(hist_size, self.W.shape[0]))
        self.y = numpy.zeros(shape=(hist_size, self.dW.shape[0]))
        self.W_1  = numpy.zeros(shape=self.W.shape)
        self.dW_1 = numpy.zeros(shape=self.dW.shape)
        self.cur_depth = 0
        # No line search is applied
        self.alpha = alpha

    def H_grad_f(self):
        q     = self.dW
        ro    = numpy.zeros(shape=self.hist_size)
        alpha = numpy.zeros(shape=self.hist_size)
        for i in range(self.cur_depth):
            ro[i]    = numpy.dot(self.y[i], self.s[i])
            alpha[i] = ro[i] * numpy.dot(self.s[i], q)
            q        = q - alpha[i] * self.y[i]
        # H_0 diagonal selection
        gamma_k = 1#numpy.dot(self.s[0], self.y[0]) / numpy.dot(self.y[0], self.y[0])
        r       = gamma_k * q
        for i in range(self.cur_depth - 1, 0, -1):
            beta = ro[i] * numpy.dot(self.y[i], r)
            r    = r + self.s[i] * (alpha[i] - beta)
        # return antigard times H
        return r

    def line_search(self, Hgf):
        pass
        #W_cur = self.W
        #for
        #    self.store()

    def step(self):
        self.load()
        # update data with new gradient
        self.s = numpy.roll(self.s, 1, axis=0)
        self.y = numpy.roll(self.y, 1, axis=0)
        self.s[0] = self.W - self.W_1
        self.y[0] = self.dW - self.dW_1
        self.W_1 = self.W
        self.dW_1 = self.dW
        # update is done now main algorithm
        Hgf = self.H_grad_f()
        if self.alpha != None: alpha = self.alpha
        else:                  alpha = self.line_search(Hgf)
        self.W = self.W - alpha * Hgf
        # incrase depth
        if self.cur_depth < self.hist_size:
            self.cur_depth += 1
        self.store()

class Sequential:
    def __init__(self, stages, optimzer):
        self.stages    = stages
        self.optimizer = optimzer(self)
    def forward(self, X, Y, predict=False):
        for stage in self.stages[0 : len(self.stages) - int(predict)]:
            X = stage.forward(X, Y)
        return X
    def backward(self):
        dY = 1
        for stage in reversed(self.stages):
            dY = stage.backward(dY)
    def update(self):
        self.optimizer.step()

class Adam(Optimizer):
    def __init__(self, model, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.0000001):
        super(Adam, self).__init__(model)
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_1_t = beta_1
        self.beta_2_t = beta_2
        self.epsilon = epsilon
        self.m_t = numpy.zeros(shape=self.W.shape)
        self.v_t = numpy.zeros(shape=self.W.shape)
        self.model = model

    def step(self):
        self.load()
        g_t = self.dW
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * g_t ** 2
        m_T = self.m_t / (1 - self.beta_1_t)
        v_T = self.v_t / (1 - self.beta_2_t)
        self.W = self.W - self.alpha * m_T / (numpy.sqrt(v_T) + self.epsilon)
        self.beta_1_t *= self.beta_1
        self.beta_2_t *= self.beta_2
        self.store()

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
              #hasattr(stage, 'W')
          L_train /= (len(X_train) / batch_size)
          # validation
          L_valid = model.forward(X_valid, Y_valid)
          print('Epoch:', epoch, 'train:', L_train, 'valid:', L_valid)

# Try on mnist data
def LoadData(size=10000):
    if 1:
        # Windows
        import _pickle as cPickle, gzip
        f = gzip.open('C:\data\mnist.pkl.gz','rb')
        _, _, test_set = cPickle.load(f, encoding='latin1')
    else:
        # Linux
        import _pickle as cPickle, gzip
        f = gzip.open('/home/zakirov/Downloads/mnist.pkl.gz', 'rb')
        _, _, test_set = cPickle.load(f, encoding='latin1')
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

cur_optimizer = 'Adam'
optimizer_setup = {'Adam' : (Adam, 32), 'SGD' : (SGD, 32), 'LBFGS' : (LBFGS, 200)}

model = Sequential([Dense(784, 50), Activation(),
                    Dense(50, 48), Activation(),
                    Dense(48, 45), Activation(),
                    Dense(45, 43), Activation(),
                    Dense(43, 40), Activation(),
                    Dense(40, 38), Activation(),
                    Dense(38, 35), Activation(),
                    Dense(35, 32), Activation(),
                    Dense(32, 30), Activation(),
                    Dense(30, 28), Activation(),
                    Dense(28, 25), Activation(),
                    Dense(25, 22), Activation(),
                    Dense(22, 20), Activation(),
                    Dense(20, 18), Activation(),
                    Dense(18, 15), Activation(),
                    Dense(15, 13), Activation(),
                    Dense(13, 10), Activation(),
                    Loss()],
                   optimizer_setup[cur_optimizer][0])
lr = Learner(model)
lr.train(X, Y, optimizer_setup[cur_optimizer][1], 100, 0.1)

R_test = model.forward(X_test, Y_test, True)
assert R_test.shape == Y_test.shape
for i in range(len(Y_test)):
    print('L:', numpy.argmax(Y_test[i]), 'R:', numpy.argmax(R_test[i]))
