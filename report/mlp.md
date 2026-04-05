# Model Comparison Study: MLP

## Model Description

Multi-Layer Perceptron (MLP) is a supervised learning algorithm and is also known as a modern feedforward neural network. It is known as being multi-layer, due to it containin an input layer, one or more hidden layers and a output layer. It consists of these fully connected dense layers that transform input data from one dimension to another. These dense layers are made up of neurons with nonlinear activation functions, which allow the network to model complex patterns and distinguish data that is not linearly separable, and slo learn new models in real-time; which can then be used for both classification or regression. 

Given a set of input features 𝑋 = {𝑥1 , 𝑥2 , … , 𝑥𝑚 }, these values are fed into the input layer of the MLP. Each neuron in the input layer represents one feature 𝑥i, so the input layer contains 𝑚 neurons. This layer forms the starting point for the data as it passes through the network’s dense layers, which can be organized into hidden layers and an output layer. There can be any number of hidden layers, and each hidden layer processes the data it receives from the input layer or the previous hidden layer. Each neuron in a hidden layer computes a weighted sum of its inputs and then applies a nonlinear activation function to determine its output, which is then passed to the next layer. Finally, the output of the last hidden layer is passed to the output layer, which produces the network’s final predictions.

The basic workflow of the MLP, can be simplifed to via forward propagation, loss functions, backpropagation and optimization.
Forward propagation is how data moves from the input layer to the output layer in a neural network. As the data passes through each layer, every neuron in a hidden or output layer combines the inputs it receives from the previous layer using a weighted sum. Specifically, each neuron computes:
\[
z = \sum_i w_i x_i + b
\]

where:  
- \(x_i\) = input value  
- \(w_i\) = corresponding weight  
- \(b\) = bias term  

After calculating the weighted sum \(z\), the neuron applies a to produce its output. This output is then passed on to the next layer in the network. Each weighted sum 'z', is then passed through an activation. Some of these activation functions include the Sigmoid, ReLU and Tanh activation functions. When we reach the output layer, the model’s error is calculated using a loss function. If the MLP is used for classification, the commonly used loss function is binary cross-entropy, while for regression, mean squared error (MSE) is typically used. Once the loss is computed, we calculate the gradient of the loss with respect to each weight, which serves as the error signal. This error is then propagated backward through the network, layer by layer, allowing us to update the weights in a way that reduces the loss in future predictions. The gradients are then used by an optimization algorithm, to updates weights and bias. Some of the most common optimizers are Stochastic Gradient Descent (SGD) and Adam Optimizer. To prevent overfitting, MLPs commonly use regularization techniques such as dropout, which randomly deactivates a fraction of neurons during training to reduce co-dependence between neurons, and L2 regularization (weight decay), which penalizes large weights in the loss function to encourage simpler, more generalizable models.


## Model Motivation
