r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1.
    **A.** The shape of the tensor is a 4D tensor [N, out_features , N, in_features] (mentioned in notebook 1 at the layer Implementations section)               
    
    
    **B.** Yes it is sparse because most of the values in the tensor are equal to 0 except for tensor[i,j,k,l] where i = k then the tensor != 0 , 
       where i = 1,2,....,N  , j = 1,2,...,out_features  , k = 1,2,..., N   ,  l = 1,2,..., in_features.
       
       Therefore we get a tensor that is mainly filled with zeros.
       the non zero elements are $\frac{d Y_(i,j)}{dX_(i,l)}$ when k = i because $Y_(i,j)$ does not depand on any other row $X_k$ where k != i . 
       where $Y_(i,j) = \sum_{k = 1}^{in_features}X_(i,k) W_(k,j)$ .                                                                                      
     
     
    **C.** no we dont have to materialize it because when we have the gradient of loss L with respect to the output tensor Y , which is 
       $\delta Y = \frac{dL }{dY}$  and we want to find the gradient of the loss with respect to X , which is 
       $\delta X = \frac{dL }{dX}$ we can use chain rule instead.
       $\delta X = \frac{dL }{dX} =  \frac{dL }{dY} * \frac{dY }{dX}$.
       $\frac{dL }{dY}$ is given so we need to calculate $\frac{d Y}{dX}$ of the output of the layer w.r.t. the input X.
       we can multiply both elements and get $\frac{dL }{dY}$.
       This approach avoids explicitly constructing the Jacobian matrix $\frac{dY }{dX}$.
       
2.
    **A.** The shape of the tensor is a 4D tensor [N, out_features , in_features, out_features] (mentioned in notebook 1 at the layer Implementations 
       section)                                                                                                                                            
    
    **B.** Yes it is sparse because most of the values in the tensor are equal to 0 except for tensor[i,j,k,l] where j = l then the tensor != 0 , 
       where i = 1,2,....,N  , j = 1,2,...,out_features  , k = 1,2,..., in_features   ,  l = 1,2,..., out_features.
       Therefore we get a tensor that is filled with many zeros.   
                                                                                                                                                          
    **C.** no we dont have to materialize it because when we have the gradient of loss L with respect to the output tensor Y , which is 
       $\delta Y = \frac{dL }{dY}$  and we want to find the gradient of the loss with respect to W , which is 
       $\delta W = \frac{dL }{dW}$ we can use chain rule instead.
       $\delta W = \frac{dL }{dw} =  \frac{dL }{dY} * \frac{dY }{dW}$.
       $\frac{dL }{dY}$ is given so we need to calculate $\frac{d Y}{dW}$ of the output of the layer w.r.t. the layer weight W.
       we can multiply both elements and get $\frac{dL }{dW}$.
       This approach avoids explicitly constructing the Jacobian matrix $\frac{dY}{dW}$.
    
"""

part1_q2 = r"""
**Your answer:**

backpropagation is essential for training neural networks with gradient-based optimization methods but not required.
Backpropagation uses the chain rule to propagate the gradient backward through the network layer by layer. This allows for efficient computation of gradients in each layer.
Direct computation of gradients without backpropagation would be computationally almost impossible , especially in deep neural networks with millions of parameters.
so in theory yes it is possible to compute gradients without backpropagation but in practice in deep neural networks it is not really possible considering really deep networks.
to sum up it is in practice required to use backpropagation in really deep neural network.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    lr = 0.05
    reg = 0.0001
    wstd = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    lr_rmsprop = 0.01
    lr_momentum = 0.01
    lr_vanilla = 0.001
    reg = 0.001
    wstd = 0.15
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**1.** High Optimization error - This usually means that there are issues with the optimization process, such as poor choice of optimization algorithm,
       learning rate, or network architecture.
       We can adjust learning rate(it might be too high or too low) , change optimization algoritm to a more sophisticated optimizers like Adam
       or modify the architecture, such as increasing the number of layers or neurons, changing activation functions.
       
       
**2.** High Generalization error - it indicates that the model performs well on the training data but poorly on unseen test data. This is typically 
       because of overfitting. Another issue might be Not enough training data.
       We can Use k-fold cross-validation to ensure the model generalizes well across different subsets of the data.
       Also using dropout, early stopping, and L2 regularization can help prevent overfitting.
       
       
**3.** High Approximation Error - This usually occurs when the model is unable to capture the relationship between input features and the target output.
       sometimes it happens when The model architecture is too basic.
       We can implement more sophisticated architectures like cnn for image data or rnn for
       sequential data, or Increase the depth of the network or width.
       In cnn, increase the receptive field to allow the network to capture larger context. This can be achieved by using larger kernels.


"""

part3_q2 = r"""
**Your answer:**

Higher FPR - Credit Card Fraud. when detecting fraudulent transactions on credit cards, a FP is done when a legitimate transaction is flagged as fraudulent, and a FN is done when a fraudulent transaction is not detected.
We expect Higher FPR because The cost of missing a fraudulent transaction (fn) might be very high for the bank. Therefore, the bank prefers to flagg legitimate transactions rather than allowing fraud to go undetected.

Higher FNR - Loan Approval Process. In the context of approving loans, a FP occurs when a loan is approved for a person who is likely to default, and a FN occurs when a loan is denied to a person who is creditworthy.
We expect Higher FNR because A conservative lending strategy might be aiming to minimize the risk of defaults, leading to a higher FNR.
The institution aims to protect its financial health by making sure that only the best applicants receive loans, even if it means turning away some potentially good customers.


"""

part3_q3 = r"""
**Your answer:**
since we aim to detect the disease early, before any symptoms appear,but Assume that these further tests are expensive and involve high-risk to the patient. Assume also that once diagnosed, a low-cost treatment exists then we would be ok with FN but make sure we minimize FP.

**1.** yes ,we still choose the same "optimal" point on the ROC curve as above. if a person with the disease develops non-lethal symptoms,then missing
       the disease early is not fatal.
       Therefore if we say that FN is when a person with the disease is classified as healthy, but he is not and FP is when a healthy person is
       classified as sick then, Since FN will eventually be noticed by symptoms, the main concern is to avoid high risk and costly tests for healthy 
       patients. Therefore, we should aim for a lower FP, accepting a higher FN.

**2.** no,we won't choose the same "optimal" point on the ROC curve as above. if a person with the disease shows no clear symptoms and may die with high
       probability if not diagnosed early enough, either by our model or by the expensive test then the main priority is to ensure that individuals with
       the disease are detected early to reduce the probability of death.
       Therefore, we should aim for a lower FN even if we get higher FP. FP are less critical. the primary concern is to survive and the cost and risk
       of further testing, though major, are less critical.

"""


part3_q4 = r"""
**Your answer:**

when dealing with text and you want to classify the sentiment of a sentence considering mlp is not the best choice since it do not consider the sequential nature of words in a sentence. mlp unlike rnn do not have a mechanism to retain context over a sequence of words. the meanning of a word in a sentence might be influenced by words around it and mlp cant retain that contex.
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn as nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

Consider the bottleneck block from the right side of the ResNet diagram above. Compare it to a regular block that performs a two 3x3 convs directly on the 256-channel input (i.e. as shown in the left side of the diagram, with a different number of channels). Explain the differences between the regular block and the bottleneck block in terms of:

Number of parameters. Calculate the exact numbers for these two examples.
Number of floating point operations required to compute an output (qualitative assessment).
Ability to combine the input: (1) spatially (within feature maps); (2) across feature maps.

**1.** 

**2.** bottelneck block : around 70k operations (256×64×1×1 ≈ 16𝐾 , 64×64×3×3 ≈ 36𝐾, 64×256×1×1 ≈ 16𝐾, Total ≈70𝐾) (lecture 3 , slide 37)
       regular block : 64 x 64 x 3 x 3 = 37k , 64 x 64 x 3 x 3 = 37k , total = 64K

**3.**

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""