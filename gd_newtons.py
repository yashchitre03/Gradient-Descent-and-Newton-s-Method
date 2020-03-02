import numpy as np
import matplotlib.pyplot as plt

w = np.array([0.49, 0.49])
eta = 0.001

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

plt.plot(all_weights[:, 0], all_weights[:, 1], '.')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Gradient Descent Weight Updates")
plt.show()

plt.plot(range(len(energy)), energy)
plt.xlabel("Number of Iterations")
plt.ylabel("Energy f(w)")
plt.title("Changes in Energy")
plt.show()



w = np.array([0.49, 0.49])
eta = 0.001

print(f"Initial Weight is: {w}\nLearning Rate is: {eta}")

all_weights = []
energy = []

H = np.empty((2, 2))


while (w[0]+w[1]) < 1 and w[0] > 0 and w[1] > 0 :
    f_w = - np.log(1 - w[0] - w[1]) - np.log(w[0]) - np.log(w[1])
    energy.append(f_w)
    
    all_weights.append(w)
    
    grad = np.array([(1 / (1-w[0]-w[1])) - (1 / w[0]), (1 / (1-w[0]-w[1])) - (1 / w[1])])
    
    H[0,:] = [(1 / (1-w[0]-w[1])**2) - (1 / w[0]**2), (1 / (1-w[0]-w[1])**2)]
    H[1,:] = [(1 / (1-w[0]-w[1])**2), (1 / (1-w[0]-w[1])**2) - (1 / w[1]**2)]
    
    delta_w = eta * np.matmul(np.linalg.inv(H), grad)
    new_w = w - delta_w
    
    if np.linalg.norm(w - new_w) < 0.00001:
        break
    else:
        w = new_w
        
        
all_weights = np.array(all_weights)
plt.plot(all_weights[:, 0], all_weights[:, 1], '.')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Newton's Method Weight Updates")
plt.show()

plt.plot(range(len(energy)), energy)
plt.xlabel("Number of Iterations")
plt.ylabel("Energy f(w)")
plt.title("Changes in Energy for Newton's Method")
plt.show()