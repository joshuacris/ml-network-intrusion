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
We selected the Multilayer Perceptron (MLP) as the neural network model in our project for two main reasons:

1. Additionally, MLP benefits directly from the scaled and log-transformed preprocessing applied to our dataset, as gradient-based optimization converges more effectively on   
  normalized inputs — an advantage that tree-based models like XGBoost and Random Forest do not require. Also given that in our preprocessing of the data, the data was scaled, MLP is a natural fit as gradient-based optimization converges more effectively on normalized inputs.

2.  MLP builds naturally on what we learned in lecture about neural networks and gradient descent, making it a fitting and well-motivated addition to our model comparison.  

## Evaluation Metrics

We evaluate MLP using the same metrics applied to all models in this study:

**ROC-AUC (Area Under the ROC Curve):** This is our primary metric for model selection and hyperparameter tuning. ROC-AUC measures the model's ability to discriminate between normal and attack traffic (0 and 1 in this binary classification problem), independent of any classification thresholds, making it robust to class imbalance. A ROC-AUC closer to 1.0 represents a better classifier.

**Precision, Recall, and F1 Score:** These metrics are evaluated at a chosen classification threshold. Precision measures what fraction of predicted attacks are true attacks (minimizing false positives), while recall measures what fraction of actual attacks were detected (minimizing false negatives). F1 score balances precision and recall. In the context of cybersecurity and network intrusion detection, false negatives (missed attacks) are more costly than false positives (flagging normal traffic as malicious), since undetected intrusions can have greater consequences.

**Threshold Tuning:** We use a default classification threshold of 0.5 as a baseline, then sweep thresholds from 0.01 to 0.99 and select the value that maximizes F1 score. This yields a better-balanced model than the default.

**Per-Attack-Category Breakdown:** We also evaluate and compare precision, recall, and F1 score separately for each of the 9 attack categories in the UNSW-NB15 dataset. This allows us to identify which attack types the model handles well versus which are difficult to detect, providing more actionable insight for future research and deployment.

## Hyperparameter Tuning
We use `RandomizedSearchCV` to tune MLP's hyperparameters, sampling 50 random configurations from the following parameter grid with 3-fold cross-validation, scored on ROC-AUC:

| Hyperparameter | Search Space |
|---|---|
| `hidden_layer_sizes` | (32,), (64,), (128,), (256,), (64,32), (128,64), (128,32), (256,128), (256,32), (128,64,32), (256,128,64), (256,128,64,32) |
| `activation` | relu, tanh |
| `learning_rate_init` | 0.01, 0.001, 0.0005, 0.0003, 0.0001 |
| `alpha` | 0.0001, 0.001, 0.01, 0.1, 1.0 |
| `batch_size` | 32, 64, 128, 256 |

The best configuration found was:

| Hyperparameter | Best Value |
|---|---|
| `hidden_layer_sizes` | (128, 64) |
| `activation` | tanh |
| `learning_rate_init` | 0.01 |
| `alpha` | 0.01 |
| `batch_size` | 128 |

**Best CV ROC-AUC: 0.9508**

## Interpretation of Results

**Overall Performance:**
The tuned MLP model achieves a test set ROC-AUC of **0.9549**, confirming reasonable discriminative ability between normal and attack traffic. The test ROC-AUC exceeds the cross-validation ROC-AUC of **0.9508**, indicating the model generalizes well to unseen data without overfitting.

**Base Evaluation (threshold = 0.5):**
At the default threshold, the model achieves high recall for the Attack class (0.95) at the cost of lower recall for Normal traffic (0.74). Attack precision is 0.70, while Normal precision is 0.96, producing an overall accuracy of 0.82. This trade-off reflects the model's tendency to predict attacks at the default threshold.           

**After Threshold Tuning:**
Sweeping the threshold and selecting the value that maximizes F1 score yields a better-balanced model, improving the trade-off between attack detection rate and false alarm rate.

**Per-Attack-Category Breakdown:**
At the tuned threshold, all attack categories achieve perfect precision (1.0), meaning no normal traffic is misclassified as any specific attack type. Recall varies substantially by category:

| Attack Type | Recall | F1 |
|---|---|---|
| Fuzzers | 0.4475 | 0.6183 |
| Shellcode | 0.9365 | 0.9672 |
| Generic | 0.9486 | 0.9736 |
| Reconnaissance | 0.9504 | 0.9746 |
| DoS | 0.9534 | 0.9762 |
| Worms | 0.9545 | 0.9767 |
| Analysis | 0.9641 | 0.9817 |
| Exploits | 0.9648 | 0.9821 |
| Backdoor | 0.9740 | 0.9868 |

**Fuzzers** remain the hardest attack type to detect (recall = 0.4475), consistent with how fuzzing generates randomized or mutated inputs that can resemble normal traffic. **Backdoor** and **Exploits** attacks are detected with near-perfect recall, likely because they produce distinctive and consistent network signatures.

