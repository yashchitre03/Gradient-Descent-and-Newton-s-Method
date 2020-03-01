import numpy as np
import matplotlib.pyplot as plt

w = np.array([0.1, 0.3])
eta = 0.01

print(w, eta)

all_weights = []
energy = []
i = 0
while (w[0]+w[1]) < 1 and w[0] > 0 and w[1] > 0 :
    i+=1
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
     
        
temp = np.array(all_weights)
plt.scatter(temp[:, 0], temp[:, 1])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Gradient Descent weight updates")
plt.show()