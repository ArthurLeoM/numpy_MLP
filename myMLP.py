import numpy as np

class Net:
    layers = []
    batch_size = 256
    input_dim = 784
    optim = "SGD"
    param_init = "norm"
    lr = 0.1
    lr_changed = False
    last_epoch = -1
    lr_scheduler = "const"
    reg = "None"
    lr_decay_idx = [5, 13, 23]

    def __init__(self, batch_size, input_dim, optim="SGD", param_init="norm", lr_scheduler="const", reg='None'):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.optim = optim
        self.param_init = param_init
        self.lr_scheduler = lr_scheduler
        self.reg = reg
        self.last_epoch = 0

    def addLinear(self, input_dim, output_dim, activation=""):
        if len(self.layers) > 0:
            assert self.layers[-1].output_dim == input_dim
        self.layers.append(Layer(input_dim, output_dim, self.batch_size, activation, self.optim, self.param_init, self.reg))
    
    def addSoftmax(self):
        self.layers.append(Softmax(self.optim))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_pred, epoch):
        grad = y_pred
        if self.lr_scheduler == "multistep":
            if epoch in self.lr_decay_idx and epoch != self.last_epoch:
                self.lr = self.lr * 0.1
                self.last_epoch = epoch
                print("Epoch %d LR decay! %.5f" % (epoch, self.lr))
        elif self.lr_scheduler == "exp":
            if epoch != self.last_epoch and self.lr >= 1e-3:
                self.lr = self.lr * 0.99
                self.last_epoch = epoch
                print("Epoch %d LR decay! %.5f" % (epoch, self.lr))

        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.lr)
        

class Layer:
    w = 0
    b = 0
    input_dim = 0
    output_dim = 0
    batch_size = 0
    activation = ""
    optim = "BGD"
    x = 0
    reg = "None"
    activated_data = 0

    def __init__(self, input_dim, output_dim, batch_size, activation, optim, param_init, reg):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.activation = activation
        self.cur_epoch = 0
        self.optim = optim
        self.reg = reg
        
        if param_init == "norm":
            self.w = np.random.normal(scale=0.01, size=(self.input_dim, self.output_dim))
            self.b = np.random.normal(scale=0.01, size=(self.output_dim))
        elif param_init == "kaiming":
            self.w = np.random.normal(scale=np.sqrt(2/self.input_dim), size=(self.input_dim, self.output_dim))
            self.b = np.random.normal(scale=np.sqrt(2/self.input_dim), size=(self.output_dim))

    def forward(self, x):
        self.x = x
        broadcast_b = np.tile(self.b, (self.x.shape[0], 1))
        z = np.dot(self.x, self.w) + broadcast_b
        if self.activation == "ReLU":
            a = (z + np.abs(z)) / 2.0
        elif self.activation == "Sigmoid":
            a = 1 / (1 + np.exp(-z))
        elif self.activation == "Tanh":
            a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif self.activation == "None":
            a = z
        self.activated_data = a
        return a

    def backward(self, grad, lr):
        y = grad
        if self.activation == "ReLU":
            self.activated_data[self.activated_data <= 0] = 0
            self.activated_data[self.activated_data > 0] = 1
            y = self.activated_data * y
        elif self.activation == "Sigmoid":
            y = self.activated_data * (1 - self.activated_data) * y
        elif self.activation == "Tanh":
            y = (1 - self.activated_data ** 2) * y
        elif self.activation == "None":
            y = 1 * y
        
        back_err = np.dot(y, self.w.T)

        if self.optim == "SGD":
            n_sample = self.x.shape[0]
            idx = np.random.randint(0, n_sample)
            w_grad = np.dot(self.x[idx].T.reshape(-1,1), y[idx].reshape(1,-1))
            b_grad = y[idx]
            if self.reg == "None":
                self.w = self.w - lr * w_grad
                self.b = self.b - lr * b_grad
            elif self.reg == 'l2':
                self.w = self.w - lr * (w_grad + 0.1 * self.w)
                self.b = self.b - lr * b_grad
            elif self.reg == 'l1':
                l1_grad = np.sign(self.w)
                l1_grad[l1_grad == 0] = 1   # make 0 grad = 1 to avoid trapping into 0
                self.w = self.w - lr * (w_grad + 0.01 * l1_grad) 
                self.b = self.b - lr * b_grad 

        elif self.optim == "BGD":
            w_grad = np.dot(self.x.T, y)
            b_grad = np.sum(y, axis=0)
            if self.reg == "None":
                self.w = self.w - lr * w_grad / self.batch_size
                self.b = self.b - lr * b_grad / self.batch_size
            elif self.reg == 'l2':
                self.w = self.w - lr * (w_grad + 0.1 * self.w) / self.batch_size
                self.b = self.b - lr * b_grad / self.batch_size
            elif self.reg == 'l1':
                l1_grad = np.sign(self.w)
                l1_grad[l1_grad == 0] = 1   # make 0 grad = 1 to avoid trapping into 0
                self.w = self.w - lr * (w_grad + 0.01 * l1_grad) / self.batch_size
                self.b = self.b - lr * b_grad / self.batch_size

        return back_err
    

class Softmax(Layer):
    y_hat = []
    optim = "BGD"
    def __init__(self, optim):
        self.optim = optim
    
    def forward(self, data):
        data = np.exp(data) 
        sum = np.sum(data, axis=1).reshape(-1, 1)
        self.y_hat = (data / sum)
        return self.y_hat

    def backward(self, y, lr):
        return self.y_hat - y


    