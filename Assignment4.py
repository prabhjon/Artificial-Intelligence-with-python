#1
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)  #
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3
plt.plot(x, y1, 'r-', label='y = 2x + 1')
plt.plot(x, y2, 'g--', label='y = 2x + 2')
plt.plot(x, y3, 'b:', label='y = 2x + 3')
plt.title("Graph of Linear Equations")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

#2
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])
plt.scatter(x, y, marker='+', color='blue', label='Data Points')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Scatter Plot of (x, y) Points")
plt.legend()
plt.show()

#3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("weight-height.csv")
length = data["Height"].values
weight = data["Weight"].values
length_cm = length * 2.54
weight_kg = weight * 0.453592
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)
print(f"Mean length (cm): {mean_length:.2f}")
print(f"Mean weight (kg): {mean_weight:.2f}")
plt.hist(length_cm, bins=20, color='blue', edgecolor='black')
plt.xlabel("Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Student Lengths")
plt.grid(True)
plt.show()

#4
import numpy as np
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
A_inv = np.linalg.inv(A)
identity1 = np.dot(A, A_inv)
identity2 = np.dot(A_inv, A)
print("A * A_inv = \n", identity1)
print("\nA_inv * A = \n", identity2)
print("\nAre both products close to the identity matrix?")
print(np.allclose(identity1, np.eye(3)))
print(np.allclose(identity2, np.eye(3)))





