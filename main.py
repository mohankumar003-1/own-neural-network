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
dense1 = Dense_Layer(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Dense_Layer(3, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_softMax_LCLE()
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)


# Let's see output of the first few samples:
print(loss_activation.output[:5])
# Print loss value
print('loss:', loss)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
# Print accuracy
print('acc:', accuracy)
# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

