import numpy as np
z = np.zeros(5)
print(z)
np.shape(z)
z2 = np.zeros((4,5)) #notice double brackets
print(z2)
np.shape(z2)
y = np.ones((2,3))
print(y)
f = np.full((7,8),11)

import numpy as np
x=np.array([[1,2,3],[3,4,5]])
print(x)
print(np.shape(x))
print(np.ndim(x))

y=np.array([[[1,2,3],[3,4,5]],[[6,7,8],[9,10,11]]])
print(y)
print(np.shape(y))
print(np.ndim(y))
print(y[1,1,0])



x = np.linspace(0,5,10)
print(x)
x2 = np.arange(0,5,0.2)
print(x2)

a = 1
b = 6
amount = 50
nopat = np.random.randint(a,b+1,amount)
x = np.random.randn(100)
print(x)
x = np.random.random(10)

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]] )
c = a + b
print(c)
d = a - b
print(d)
e = a / b
print(e)
x = a * b
print(x)
f = np.sin(a)
print(f)
g = np.cos(a)
print(a)
h = np.tan(a)
print(h)
i = np.sqrt(a)
print(i)
j = np.dot(a,b)
print(j)
k = a>=1
print(k)
l = a <1
print(l)

import numpy as np

n = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
n = np.reshape(n, (3,4))
print(n)

new1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
new2 = new1.reshape(4,3)
print(new2)
print(new2[:,0])

import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]] )
c = a + b
print(c)
d = a - b
print(d)
e = a / b
print(e)
x = a * b
print(x)
f = np.sin(a)
print(f)
g = np.cos(a)
print(a)
h = np.tan(a)
print(h)
i = np.sqrt(a)
print(i)
j = np.dot(a,b)
print(j)
k = a>=1
print(k)
l = a <1
print(l)


