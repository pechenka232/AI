#реализует градиентный спуск с импульсом для минимизации функции f(x)=x**2-5*x+5, прекращая работу когда изменение y становится меньше epsilon
import numpy as np

def f(x):
    return x**2 - 5*x + 5

def grad_f(x):
    return 2*x - 5


alpha = 0.01
beta = 0.5
num_steps = 1000
epsilon = 1e-5


x_current = 0
velocity = 0
y_vals = [f(x_current)]
delta_y = None


for step in range(num_steps):
  
    grad = grad_f(x_current)
    
 
    velocity = beta * velocity + alpha * grad
    
   
    x_next = x_current - velocity
    y_next = f(x_next)
    
   
    y_prev = y_vals[-1]
    delta_y = abs(y_next - y_prev)
    
    
    y_vals.append(y_next)
    

    if delta_y < epsilon:
        x_current = x_next 
        break
        

    x_current = x_next


print(delta_y)
