import numpy as np
import time, os


n = 10         
win = 9        
steps = 2    
eps = 0.9      # шанс чето потыкать случайно
lr = 0.1       
gamma = 0.9    


q = np.zeros((n, steps))



for ep in range(100): #сколько времени вы хотите дать болванчикy на обyчение
    pos = 0
    done = False
    cnt = 0
    
    while not done:
        
        if np.random.rand() > eps or np.all(q[pos] == 0):
            act = np.random.randint(0, 2)
        else:
            act = np.argmax(q[pos])
            
        old_pos = pos
        
        
        if act == 1 and pos < n - 1:
            pos += 1
        elif act == 0 and pos > 0:
            pos -= 1
            
        priz = 1 if pos == win else 0
        if priz == 1: done = True
            
        q[old_pos, act] += lr * (priz + gamma * np.max(q[pos]) - q[old_pos, act])
        
       
        m = ['-'] * n
        m[pos] = '🤖'
        os.system('cls||clear')
        print(f"Попытка: {ep+1} | Шагов: {cnt}")
        print("".join(m) + " ⭐")
        time.sleep(0.05)
        cnt += 1

print(" Таблица весов:")
print(q)
