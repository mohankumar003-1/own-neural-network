import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

X,y = spiral_data(100,3)
class Dense_Layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights  = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

class ReLU:
     def forward(self,inputs):
         self.output = np.maximum(0,inputs)
class softMax:
     def forward(self,inputs):
         expValues = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
         probability = expValues / np.sum(expValues,axis=1,keepdims=True)
         self.output = probability
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
x = [[1,2,3,4],[0.5,8,-5.0,1],[2,6.0,-0.2,0.9]]

layer1 = Dense_Layer(2,3)
activation = ReLU()
layer1.forward(X)
activation.forward(layer1.output)
softmax = softMax()
softmax.forward(activation.output)
print(softmax.output[:5])
lossFunction = Loss_CLE()
loss = lossFunction.calculate(softmax.output , y)

print(loss)
