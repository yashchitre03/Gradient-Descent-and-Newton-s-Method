# Gradient Descent and Newton's Optimization Method

This python program implements the Gradient Descent as well as the Newton's approach to find their performance on a random dataset to reduce the error in order to fit the dataset. The plots can be seen by running the jupyter notebook.

## Comparing the speed of convergence of gradient descent and Newton’s method:

For the weights chosen, we can see that the Newton's method requires more iterations to converge as compared to the gradient descent method.

But, we can find certain weights, and learning rate that makes the Newton's method converge faster. It may be because in general the Newton's method converges faster.

Hence, we cannot generalize as such and certain initial weights, and learning rate may benefit the gradient descent, while certain learning rate may be optimal for the Newton's approach.

### Refer:

1. [Gradient Descent Method](https://en.wikipedia.org/wiki/Gradient_descent)
2. [Newton's Optomization Method](https://en.wikipedia.org/wiki/Newton's_method_in_optimization)

### Libraries Used:

1. Numpy library is used to store and manipulate the data.

2. Matplotlib is used in order to plot the results.