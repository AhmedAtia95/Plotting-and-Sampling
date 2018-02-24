
# coding: utf-8

# 
# # Import Libraries

# import matplotlib.pyplot as plt
# %matplotlib inline
# import numpy as np
# from numpy import sin, cos, pi
# from numpy.random import normal,rand

# # Coding 

# # 1-Exponential

# In[9]:


a = np.linspace(0,10,100)
b = np.exp(a)
plt.plot(a,b)
plt.title("+ve exp")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()


# # Sampling

# In[33]:


A=2
f=50.0
T=1/f

fs = 2*f
Ts = 1/f

cycles=10
t=np.arange(0,cycles*T,Ts)
x=A*np.exp(2*np.pi*f*t)
plt.plot(x)
plt.show()


#  # 2- Sine
# 

# In[42]:



import numpy, matplotlib
from numpy import sin, cos, pi
from matplotlib import pyplot as plt

x = numpy.linspace(-pi,pi,100)
ysin=sin(x)

def Create_plot(c,v,n):
    plt.plot(c,v)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend((n))
    plt.title('Plot of sin(x) from -pi to pi')
    plt.show()
    
     
Create_plot(x,ysin,'sin(x)')


# #    Sampling

# In[30]:


A=2
f=50.0
T=1/f

fs = 2*f
Ts = 1/f

cycles=10
t=np.arange(0,cycles*T,Ts)
x=A*np.sin(2*np.pi*f*t)
plt.plot(x)
plt.show()


# # 3- Cosine

# In[39]:


import numpy, matplotlib
from numpy import sin, cos, pi
from matplotlib import pyplot as plt

x = numpy.linspace(-pi,pi,100)
ycos=cos(x)

def Create_plot(c,v,n):
    plt.plot(c,v)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend((n))
    plt.title('Plot of cos(x) from -pi to pi')
    plt.show()
    
     
Create_plot(x,ycos,'cos(x)')


# #    Sampling

# In[44]:


A=2
f=50.0
T=1/f

fs = 2*f
Ts = 1/f

cycles=10
t=np.arange(0,cycles*T,Ts)
x=A*np.cos(2*np.pi*f*t)
plt.plot(x)
plt.show()

