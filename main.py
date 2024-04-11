import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

X,y = spiral_data(100,3)
class Dense_Layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights  = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.inputs = n_inputs
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights)+self.biases
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis = 0 ,keepdims = True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class ReLU:
     def forward(self,inputs):
         self.inputs = inputs
         self.output = np.maximum(0,inputs)
     def backward(self,dvalues):
         self.dinputs = dvalues.copy()
         self.dinputs[self.inputs<=0] = 0
class softMax:
     def forward(self,inputs):
         self.inputs = inputs
         expValues = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
         probability = expValues / np.sum(expValues,axis=1,keepdims=True)
         self.output = probability
     def backward(self,dvalues):
         self.dinputs = np.empty_like(dvalues)
         for index,(single_output,single_dvalues) in  enumerate(zip(self.output,dvalues)):
             single_output = single_output.reshape(-1,1)
             jacobian_matrix = np.diagflat(single_output) -  np.dot(single_output,single_output.T)
             self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
class Loss:
    def calculate(self, output , y):
        sampleLoss = self.forward(output,y)
        data_loss  = np.mean(sampleLoss)
        return data_loss
class Loss_CLE(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1 :
            correct_confidence = y_pred_clip[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clip * y_true , axis=1)
        neg_log = -np.log(correct_confidence)
        return neg_log
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples
class Activation_softMax_LCLE():
    def __init__(self):
        self.activation = softMax()
        self.loss = Loss_CLE()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2 :
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs / samples
class Optimizer_SGD:
    def __init__(self,learning_rate=1.0,decay=0.,momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1.0/(1.0 + (self.decay * self.iterations)))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weights = weights - self.learning_rate * layer.dweights
            biases = biases - self.learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *  (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate *  layer.dbiases /(np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
'''softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])
softmax_loss = Activation_softMax_LCLE()
softmax_loss.backward(softmax_outputs,class_targets)
dvalues1 = softmax_loss.dinputs

activation = softMax()
activation.output = softmax_outputs
loss = Loss_CLE()
loss.backward(softmax_outputs,class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print(dvalues1)
print(dvalues2)'''
dense1 = Dense_Layer(2, 64)
activation1 = ReLU()
dense2 = Dense_Layer(64, 3)
loss_activation = Activation_softMax_LCLE()
optimizer = Optimizer_Adam(learning_rate=0.02,decay=1e-5)
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)


    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch%100:
        print('epoch: %d , acc: %.3f , loss: %.3f'%(epoch,accuracy,loss))
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

