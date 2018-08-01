import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== y_data = b+ w * x_data ======
x_data = [338, 333, 328, 207, 226, 25, 179, 60, 208, 606]
y_data = [640, 633, 619, 393, 428, 27, 193, 66, 226, 1591]
df = pd.DataFrame(np.array([x_data, y_data]).T, columns=['x', 'y'])


# ====== Get grid of ws, bs and errors ======
bs = np.arange(-200, -99, 1)
ws = np.arange(-5, 5.1, 0.1)
b_grid, w_grid = np.meshgrid(bs, ws)
Z = np.zeros((len(bs), len(ws)))
for x in range(len(bs)):
    for y in range(len(ws)):
        Z[y][x] = ((df['y'] - (bs[x] + ws[y]*df['x']))**2).mean()
plt.contourf(bs, ws, Z, 30, alpha=0.5)
plt.show()

# ====== Gradient Descent ======
# --- initialization ---
b = -120                # initialization
w = -4                  # initialization
iteration = 10000       # maximum steps in gradient descent
lr = 3                  # learning rate
lr_b = 0                # updating learning rate for b with factor lr_b
lr_w = 0                # updating learning rate for w with factor lr_w

# --- store historical (b,w) in lists ---
b_hist = [b]
w_hist = [w]

# --- gradient descent ---
for k in range(iteration):
    # a. calculate gradient
    se = df['y'] - b - w * df['x']
    b_grad = - 2 * se.sum()
    w_grad = - 2 * (se * df['x']).sum()
    # b. update learning factors
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    # c. update parameters
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad
    # d. store historical parameters
    b_hist.append(b)
    w_hist.append(w)


# ====== plot the final result ======
plt.contourf(bs, ws, Z, 30, alpha=0.5)
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_hist, w_hist, 'o-', ms=5, lw=1.5, color='k')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$alpha$', fontsize=16)
plt.xlabel(r'$w$', fontsize=16)
plt.show()        
        
