import numpy as np
import matplotlib.pyplot as plt

w = np.array([0.8, 0.1])
eta = 0.007

print(w, eta)

all_weights = []
energy = []

while (w[0]+w[1]) < 1 and w[0] > 0 and w[1] > 0 :
    f_w = - np.log(1 - w[0] - w[1]) - np.log(w[0]) - np.log(w[1])
    energy.append(f_w)
    
    all_weights.append(w)
    
    grad = np.array([(1 / (1-w[0]-w[1])) - (1 / w[0]), (1 / (1-w[0]-w[1])) - (1 / w[1])])
    
    delta_w = eta * grad
    new_w = w - delta_w
    
    if np.linalg.norm(w - new_w) < 0.001:
        break
    else:
        w = new_w
     
all_weights = np.array(all_weights)

plt.scatter(all_weights[:, 0], all_weights[:, 1])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Gradient Descent Weight Updates")
plt.show()

plt.scatter(range(len(energy)), energy)
plt.xlabel("Number of Iterations")
plt.ylabel("Energy f(w)")
plt.title("Changes in Energy")
plt.show()
