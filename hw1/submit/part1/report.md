---
header: Deep Learning (84100343-0)
title: "Part One: Backpropogation"
---

# Part One: Backpropogation

## Task 1

> Given a softmax function, please calculate the gradients of **the output of a softmax function** with respect to **its input**. (**5 points**)

To calculate the gradients of the output of a softmax function with respect to its input, let's start by defining the softmax function and its properties.

### Softmax Function

Given an input vector $\mathbf{z} = [z_1, z_2, \dots, z_n]^T$, the softmax function is defined as:

```math
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
```

for $i = 1, 2, \dots, n$. The output of the softmax function is a probability distribution over $n$ classes, denoted as $\mathbf{y} = [y_1, y_2, \dots, y_n]^T$, where:

```math
y_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
```

### Gradient Calculation

We want to compute the gradient of the softmax output $\mathbf{y}$ with respect to the input $\mathbf{z}$, i.e., $\frac{\partial \mathbf{y}}{\partial \mathbf{z}}$.

#### Step-by-Step Derivation

1. **Single Element Gradient**:
   Let's first compute the gradient of a single output element $y_i$ with respect to a single input element $z_k$:
   ```math
   \frac{\partial y_i}{\partial z_k}
   ```
   There are two cases to consider:
   - **Case 1: $i = k$**
     ```math
     \frac{\partial y_i}{\partial z_i} = \frac{\partial}{\partial z_i} \left( \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \right)
     ```
     Using the quotient rule:
     ```math
     \frac{\partial y_i}{\partial z_i} = \frac{e^{z_i} \cdot \sum_{j=1}^n e^{z_j} - e^{z_i} \cdot e^{z_i}}{(\sum_{j=1}^n e^{z_j})^2} = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} - \left( \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \right)^2 = y_i (1 - y_i)
     ```
   - **Case 2: $i \neq k$**
     ```math
     \frac{\partial y_i}{\partial z_k} = \frac{\partial}{\partial z_k} \left( \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \right)
     ```
     Again, using the quotient rule:
     ```math
     \frac{\partial y_i}{\partial z_k} = \frac{0 - e^{z_i} \cdot e^{z_k}}{(\sum_{j=1}^n e^{z_j})^2} = - \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \cdot \frac{e^{z_k}}{\sum_{j=1}^n e^{z_j}} = - y_i y_k
     ```
2. **Vectorial Form**:
   Combining the two cases, we can write the gradient in vectorial form. Let $\mathbf{y} = \text{softmax}(\mathbf{z})$ and $\mathbf{z} = [z_1, z_2, \dots, z_n]^T$. The gradient $\frac{\partial \mathbf{y}}{\partial \mathbf{z}}$ is a Jacobian matrix $J$ where:
   ```math
   J_{ij} = \frac{\partial y_i}{\partial z_j}
   ```
   From the derivations above:
   ```math
   J_{ij} = \begin{cases}
   y_i (1 - y_i) & \text{if } i = j \\
   - y_i y_j & \text{if } i \neq j
   \end{cases}
   ```
   Therefore, the Jacobian matrix $J$ can be written as:
   ```math
   J = \text{diag}(\mathbf{y}) - \mathbf{y} \mathbf{y}^T
   ```
   where $\text{diag}(\mathbf{y})$ is a diagonal matrix with the elements of $\mathbf{y}$ on the diagonal.

### Summary

The gradient of the softmax output $\mathbf{y}$ with respect to its input $\mathbf{z}$ is given by the Jacobian matrix:

```math
\frac{\partial \mathbf{y}}{\partial \mathbf{z}} = \text{diag}(\mathbf{y}) - \mathbf{y} \mathbf{y}^T
```

This matrix represents the sensitivity of each output element $y_i$ to each input element $z_j$, encapsulating the local gradient information necessary for backpropagation in neural networks.

## Task 2

> Finish the detailed **feed-forward computations** of a batch of samples $(\mathbf{X}, \mathbf{Y})$ during a training iteration, coming with final predictions $\mathbf{\hat{Y}}$. (**10 points**)

### Detailed Feed-Forward Computations

Given a mini-batch of training samples $(\mathbf{X}, \mathbf{Y})$ with a batch size of $m$, where each input sample $\mathbf{X}^i \in \mathbb{R}^{L \times D}$ and each target $\mathbf{Y}^i$ is a one-hot vector for classification with $K$ classes, we will perform the detailed feed-forward computations to obtain the final predictions $\mathbf{\hat{Y}}$.

#### Step-by-Step Feed-Forward Computation

1. **Input Layer:**
   - For each sample $\mathbf{X}^i \in \mathbb{R}^{L \times D}$, the input remains $\mathbf{X}^i$.
2. **Transpose Operation (1st Transpose):**
   - Transpose the input matrix $\mathbf{X}^i$:
     ```math
     \mathbf{H}_1 = {\mathbf{X}^i}^T \in \mathbb{R}^{D \times L}
     ```
3. **Fully Connected Layer 1 (FC-1) with ReLU Activation:**
   - Apply the first fully connected layer with weights $\mathbf{\Theta}_1 \in \mathbb{R}^{L \times L}$ and bias $\mathbf{b}_1 \in \mathbb{R}^L$:
     ```math
     \mathbf{H}_2 = \mathrm{ReLU}({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1) \in \mathbb{R}^{D \times L}
     ```
   - Here, $\mathbf{1}_D$ is a vector of ones with dimension $D$.
4. **Transpose Operation (2nd Transpose):**
   - Transpose the output of FC-1:
     ```math
     \mathbf{H}_3 = \mathbf{H}_2^T = \mathrm{ReLU}(\mathbf{\Theta}_1^T \mathbf{X}^i + \mathbf{b}_1 \otimes \mathbf{1}_D) \in \mathbb{R}^{L \times D}
     ```
5. **Skip-Connection:**
   - Add the original input $\mathbf{X}^i$ to the output of the second transpose:
     ```math
     \mathbf{H}_4 = \mathbf{H}_3 + \mathbf{X}^i \in \mathbb{R}^{L \times D}
     ```
6. **Fully Connected Layer 2 (FC-2) with ReLU Activation:**
   - Apply the second fully connected layer with weights $\mathbf{\Theta}_2 \in \mathbb{R}^{D \times D}$ and bias $\mathbf{b}_2 \in \mathbb{R}^D$:
     ```math
     \mathbf{H}_5 = \mathrm{ReLU}(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2) \in \mathbb{R}^{L \times D}
     ```
   - Here, $\mathbf{1}_L$ is a vector of ones with dimension $L$.
7. **Mean Operation:**
   - Compute the mean along the feature dimension:
     ```math
     \mathbf{H}_6 = \frac{1}{D} \mathbf{H}_5 \mathbf{1}_D \in \mathbb{R}^L
     ```
8. **Fully Connected Layer 3 (FC-3) with Softmax Activation:**
   - Apply the third fully connected layer with weights $\mathbf{\Theta}_3 \in \mathbb{R}^{L \times K}$ and bias $\mathbf{b}_3 \in \mathbb{R}^K$:
     ```math
     \mathbf{\hat{Y}}^i = \mathrm{Softmax}(\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3) \in \mathbb{R}^K
     ```

#### Final Predictions

For each sample $\mathbf{X}^i$ in the mini-batch, the final prediction $\mathbf{\hat{Y}}^i$ is obtained through the above feed-forward computations. The final predictions for the entire mini-batch are:

```math
\mathbf{\hat{Y}} = \{\mathbf{\hat{Y}}^1, \mathbf{\hat{Y}}^2, \dots, \mathbf{\hat{Y}}^m\}
```

where each $\mathbf{\hat{Y}}^i \in \mathbb{R}^K$ is the predicted probability distribution over the $K$ classes for the $i$-th sample.

This completes the detailed feed-forward computations for a batch of samples $(\mathbf{X}, \mathbf{Y})$ during a training iteration, resulting in the final predictions $\mathbf{\hat{Y}}$.

## Task 3

> Use the backpropagation algorithm we have learned in class and give **the gradients of the overall loss in a mini-batch with respect to the parameters at each layer**. (**15 points**)

To compute the gradients of the overall loss with respect to the parameters at each layer, we will use the backpropagation algorithm. We will start from the final layer and propagate the gradients backward through the network.

### Notations

- $\mathbf{X}^i \in \mathbb{R}^{L \times D}$: Input sample.
- $\mathbf{Y}^i \in \mathbb{R}^K$: One-hot encoded target vector.
- $\mathbf{\hat{Y}}^i \in \mathbb{R}^K$: Predicted output vector.
- $\mathbf{\Theta}_1, \mathbf{\Theta}_2, \mathbf{\Theta}_3$: Weight matrices for FC-1, FC-2, and FC-3.
- $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$: Bias vectors for FC-1, FC-2, and FC-3.
- $\mathbf{H}_1, \mathbf{H}_2, \dots, \mathbf{H}_6$: Intermediate outputs at each layer.

### Loss Function

The cross-entropy loss for a mini-batch of size $m$ is:

```math
\mathcal{L} = \frac{1}{m} \sum_{i=1}^m \left[ - \sum_{k=1}^K \mathbf{Y}_k^i \log{\mathbf{\hat{Y}}_k^i} \right]
```

### Backpropagation

#### Step 1: Gradient of Loss with Respect to $\mathbf{\hat{Y}}^i$

The gradient of the loss with respect to the predicted output $\mathbf{\hat{Y}}^i$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{\hat{Y}}^i} = \frac{1}{m} \left( - \frac{\mathbf{Y}^i}{\mathbf{\hat{Y}}^i} \right)
```

#### Step 2: Gradient of Loss with Respect to $\mathbf{H}_6$

The output of the softmax function is $\mathbf{\hat{Y}}^i = \text{Softmax}(\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)$. Using the Jacobian matrix derived in Task 1:

```math
\frac{\partial \mathbf{\hat{Y}}^i}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)} = \text{diag}(\mathbf{\hat{Y}}^i) - \mathbf{\hat{Y}}^i {\mathbf{\hat{Y}}^i}^T
```

Thus,

```math
\frac{\partial \mathcal{L}}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)} = \frac{\partial \mathcal{L}}{\partial \mathbf{\hat{Y}}^i} \cdot \left( \text{diag}(\mathbf{\hat{Y}}^i) - \mathbf{\hat{Y}}^i {\mathbf{\hat{Y}}^i}^T \right)
```

Since $\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3$ is a vector, we can write:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_6} = \frac{\partial \mathcal{L}}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)} \cdot \mathbf{\Theta}_3^T
```

#### Step 3: Gradient of Loss with Respect to $\mathbf{\Theta}_3$ and $\mathbf{b}_3$

The gradient of the loss with respect to the weights $\mathbf{\Theta}_3$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_3} = \mathbf{H}_6^T \cdot \frac{\partial \mathcal{L}}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)}
```

The gradient of the loss with respect to the bias $\mathbf{b}_3$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_3} = \frac{\partial \mathcal{L}}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)}
```

#### Step 4: Gradient of Loss with Respect to $\mathbf{H}_5$

The output of the mean operation is $\mathbf{H}_6 = \frac{1}{D} \mathbf{H}_5 \mathbf{1}_D$. The gradient of the loss with respect to $\mathbf{H}_5$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_5} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}_6} \cdot \frac{1}{D} \mathbf{1}_D^T
```

#### Step 5: Gradient of Loss with Respect to $\mathbf{H}_4$

The output of FC-2 is $\mathbf{H}_5 = \text{ReLU}(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2)$. The gradient of the loss with respect to $\mathbf{H}_4$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_4} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}_5} \cdot \mathbf{\Theta}_2^T \odot \text{ReLU}'(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2)
```

where $\odot$ denotes element-wise multiplication and $\text{ReLU}'$ is the derivative of the ReLU function.

#### Step 6: Gradient of Loss with Respect to $\mathbf{\Theta}_2$ and $\mathbf{b}_2$

The gradient of the loss with respect to the weights $\mathbf{\Theta}_2$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_2} = \mathbf{H}_4^T \cdot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_5} \odot \text{ReLU}'(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2) \right)
```

The gradient of the loss with respect to the bias $\mathbf{b}_2$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \sum_{i=1}^L \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_5} \odot \text{ReLU}'(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2) \right)
```

#### Step 7: Gradient of Loss with Respect to $\mathbf{H}_3$

The output of the skip-connection is $\mathbf{H}_4 = \mathbf{H}_3 + \mathbf{X}^i$. The gradient of the loss with respect to $\mathbf{H}_3$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_3} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}_4}
```

#### Step 8: Gradient of Loss with Respect to $\mathbf{H}_2$

The output of the second transpose is $\mathbf{H}_3 = \mathbf{H}_2^T$. The gradient of the loss with respect to $\mathbf{H}_2$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_2} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_3} \right)^T
```

#### Step 9: Gradient of Loss with Respect to $\mathbf{H}_1$

The output of FC-1 is $\mathbf{H}_2 = \text{ReLU}({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1)$. The gradient of the loss with respect to $\mathbf{H}_1$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}_2} \cdot \mathbf{\Theta}_1^T \odot \text{ReLU}'({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1)
```

#### Step 10: Gradient of Loss with Respect to $\mathbf{\Theta}_1$ and $\mathbf{b}_1$

The gradient of the loss with respect to the weights $\mathbf{\Theta}_1$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_1} = {\mathbf{X}^i}^T \cdot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_2} \odot \text{ReLU}'({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1) \right)
```

The gradient of the loss with respect to the bias $\mathbf{b}_1$ is:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum_{i=1}^D \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_2} \odot \text{ReLU}'({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1) \right)
```

### Summary of Gradients

- **FC-3:**
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_3} = \mathbf{H}_6^T \cdot \frac{\partial \mathcal{L}}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)}$
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_3} = \frac{\partial \mathcal{L}}{\partial (\mathbf{H}_6 \mathbf{\Theta}_3 + \mathbf{b}_3)}$
- **FC-2:**
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_2} = \mathbf{H}_4^T \cdot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_5} \odot \text{ReLU}'(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2) \right)$
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \sum_{i=1}^L \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_5} \odot \text{ReLU}'(\mathbf{H}_4 \mathbf{\Theta}_2 + \mathbf{1}_L \otimes \mathbf{b}_2) \right)$
- **FC-1:**
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_1} = {\mathbf{X}^i}^T \cdot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_2} \odot \text{ReLU}'({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1) \right)$
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum_{i=1}^D \left( \frac{\partial \mathcal{L}}{\partial \mathbf{H}_2} \odot \text{ReLU}'({\mathbf{X}^i}^T \mathbf{\Theta}_1 + \mathbf{1}_D \otimes \mathbf{b}_1) \right)$

These gradients are computed for each sample in the mini-batch and then averaged to obtain the final gradients for the mini-batch.

## Task 4

> Write the pseudo-code for the stochastic gradient descent (SGD) algorithm to update the parameters at each layer during training. (**5 points**)

```python
# Initialize parameters
Theta_1 = initialize_weights(L, L)
Theta_2 = initialize_weights(D, D)
Theta_3 = initialize_weights(L, K)
b_1 = initialize_bias(L)
b_2 = initialize_bias(D)
b_3 = initialize_bias(K)

# Initialize learning rate
eta = 0.01

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        X_batch, Y_batch = batch

        # Forward pass to compute predictions and loss
        H_1 = transpose(X_batch)
        H_2 = ReLU(H_1 @ Theta_1 + ones_D @ b_1)
        H_3 = transpose(H_2)
        H_4 = H_3 + X_batch
        H_5 = ReLU(H_4 @ Theta_2 + ones_L @ b_2)
        H_6 = mean(H_5, axis=1)
        Y_hat = softmax(H_6 @ Theta_3 + b_3)
        loss = cross_entropy_loss(Y_batch, Y_hat)

        # Backward pass to compute gradients
        dL_dY_hat = (1 / m) * (-Y_batch / Y_hat)
        dL_dH_6_Theta_3_b_3 = dL_dY_hat @ (diag(Y_hat) - Y_hat @ Y_hat.T)
        dL_dH_6 = dL_dH_6_Theta_3_b_3 @ Theta_3.T
        dL_dTheta_3 = H_6.T @ dL_dH_6_Theta_3_b_3
        dL_db_3 = sum(dL_dH_6_Theta_3_b_3, axis=0)

        dL_dH_5 = dL_dH_6 @ (1 / D) * ones_D.T
        dL_dH_4 = dL_dH_5 @ Theta_2.T * ReLU_prime(H_4 @ Theta_2 + ones_L @ b_2)
        dL_dTheta_2 = H_4.T @ (dL_dH_5 * ReLU_prime(H_4 @ Theta_2 + ones_L @ b_2))
        dL_db_2 = sum(dL_dH_5 * ReLU_prime(H_4 @ Theta_2 + ones_L @ b_2), axis=0)

        dL_dH_3 = dL_dH_4
        dL_dH_2 = transpose(dL_dH_3)
        dL_dH_1 = dL_dH_2 @ Theta_1.T * ReLU_prime(H_1 @ Theta_1 + ones_D @ b_1)
        dL_dTheta_1 = H_1.T @ (dL_dH_2 * ReLU_prime(H_1 @ Theta_1 + ones_D @ b_1))
        dL_db_1 = sum(dL_dH_2 * ReLU_prime(H_1 @ Theta_1 + ones_D @ b_1), axis=0)

        # Update parameters
        Theta_1 -= eta * dL_dTheta_1
        Theta_2 -= eta * dL_dTheta_2
        Theta_3 -= eta * dL_dTheta_3
        b_1 -= eta * dL_db_1
        b_2 -= eta * dL_db_2
        b_3 -= eta * dL_db_3
```
