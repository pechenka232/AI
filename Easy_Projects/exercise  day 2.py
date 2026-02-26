import numpy as np

# f(x) = x^2 - 5x + 5
def graf_f(x):
    return 2 * x - 5


alpha = 0.01
beta = 0.5
pos = 0       
velocity = 0   

gradient = graf_f(pos)
velocity = beta * velocity + alpha * gradient
x_next = pos - velocity

print(round(x_next, 5))



#######################
import numpy as np

#Функция и её градиент 
def f(x):
    return x**2 - 5*x + 5

def graf_f(x):
    return 2*x - 5


x = 0.0          
alpha = 0.01     
beta = 0.5       #параметры
epsilon = 1e-5   
velocity = 0.0  
max_iter = 1000  
steps = 0

for i in range(max_iter):
    f_old = f(x)  
    gradient = graf_f(x)
    velocity = beta * velocity + alpha * gradient
  
    x = x - velocity
    steps += 1

  
    f_new = f(x)
    if abs(f_new - f_old) < epsilon:
        break
print(steps)
