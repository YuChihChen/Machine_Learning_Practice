import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('\n============ I. Generate and plot Data ============')
size  = 100
x = 2 * np.random.rand(size)
y_true = 4 + 3 * x
y = y_true + np.random.randn(size)
plt.figure(figsize=(10, 8))
plt.scatter(x, y)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
df = pd.DataFrame(np.array([np.ones((size)), x, y]).T, columns=['x0', 'x1', 'y'])
print('Exact Solution: y = b0 + b1 * x1 + noise, where b0 = 4, b1 = 3')



print('\n============ II. Linear regression using the normal equation ============')
X = df.iloc[:, 0:2].values.copy()
y = df['y'].reshape(size, 1)
theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X.dot(np.array(theta_best))
plt.figure(figsize=(10, 8))
plt.plot(x, y_true, "g-", label='true')
plt.plot(x, y_pred, "r-", label='fitted')
plt.scatter(x, y, label='data')
plt.axis([0, 2, 0, 15])
plt.legend()
plt.show()
print('(b0, b1) = {}'.format(theta_best.T[0]))



print('\n============ III. Linear regression using gradient descent ============')
# ------ 1. batch gradient descent ------
eta = 0.1
n_iterations = 1000
theta_path_bgd = list()
theta = np.random.randn(2,1)
theta0 = theta
for iteration in range(n_iterations):
    gradients = (2 / size) * X.T.dot(X.dot(theta) - y)
    theta = theta - eta * gradients
    theta_path_bgd.append(theta.T[0])
print('batch      gradient descent: (b0, b1) = {}'.format(theta.T[0]))


# ------ 2. stochastic gradient descent ------
def learning_schedule(t, t0, t1):
    return t0 / (t + t1)

theta_path_sgd = list()
n_epochs = 50
theta = theta0
for epoch in range(n_epochs):
    for i in range(size):
        random_index = np.random.randint(size)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * size + i, t0=5, t1=50)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta.T[0])  
print('stochastic gradient descent: (b0, b1) = {}'.format(theta.T[0]))


# ------ 3. mini batch gradient descent ------
theta_path_mgd = list()
n_iterations = 50
minibatch_size = 25
theta = theta0
t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(size)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, size, minibatch_size):
        t += 1
        xi = X_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t, 200, 1000)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
print('mini-batch gradient descent: (b0, b1) = {}'.format(theta.T[0]))

# ------ 4. plots ------
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)
plt.figure(figsize=(10,8))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()

