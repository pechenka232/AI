import numpy as np
import matplotlib.pyplot as plt

def generate_market_data():
    X = 2 * np.random.rand(100, 1)
    
    # Случайный коэффициент цены и шум (чтобы точки не лежали на идеальной линии)
    true_w = np.random.randint(5, 50) 
    true_b = np.random.randint(2, 10)
    
    # Формула: y = w*X + b + случайный шум
    y = true_b + true_w * X + np.random.randn(100, 1) * 2
    
    return X, y, true_w, true_b


class LinearBrain:
    def __init__(self):
        
        self.w = np.random.randn(1, 1)
        self.b = np.random.randn(1, 1)

    def predict(self, X):
        return X.dot(self.w) + self.b

    def train(self, X, y, alpha=0.1, epochs=100):
        
        for epoch in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            
            
            grad_w = (2/len(X)) * X.T.dot(error)
            grad_b = (2/len(X)) * np.sum(error)
            
            # w_new = w_old - alpha * gradient
            self.w -= alpha * grad_w
            self.b -= alpha * grad_b


X, y, real_w, real_b = generate_market_data()
brain = LinearBrain()

print(f"Реальный коэффициент : {real_w}")
brain.train(X, y, alpha=0.1, epochs=200)
print(f"ИИ вычислил коэффициент: {brain.w[0][0]:.2f}")






#визyализация
plt.scatter(X, y, color='blue', label='Реальные данные (рынок)')
plt.plot(X, brain.predict(X), color='red', linewidth=3, label='Прогноз ИИ (регрессия)')
plt.title(f"ИИ нашел зависимость: y = {brain.w[0][0]:.1f}x + {brain.b[0][0]:.1f}")
plt.xlabel("Износ (Float)")
plt.ylabel("Цена ($)")
plt.legend()
plt.grid(True)
plt.show()
