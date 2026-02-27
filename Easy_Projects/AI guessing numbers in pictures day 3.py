import torch #выгрyжай картинкy num.png(в папке проекта) с цифрой от 0-9 ИИ постарается yгадать что это за число  (day 3)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.RandomRotation(30),       
    transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)), 
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class SuperConvNet(nn.Module):
    def __init__(self):
        super(SuperConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.drop_out = nn.Dropout(0.3) # выключение 30 процентов нейронов(чтобы ИИ искала отличительные признаки )
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop_out(out)
        out = torch.relu(self.fc1(out))
        return self.fc2(out)


MODEL = "ai_num.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuperConvNet().to(device)


if not os.path.exists(MODEL):
    print("Загрузка датасеты(МНИСТ)")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X_tensor = torch.FloatTensor(X_raw).reshape(-1, 1, 28, 28) / 255.0
    y_tensor = torch.LongTensor(y_raw.astype(int))
    
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Обучение с вращением и сдвигами")
    model.train()
    for epoch in range(10): 
        for images, labels in loader:
           
            shift_x, shift_y = np.random.randint(-5, 5, 2)
            images = torch.roll(images, shifts=(shift_y, shift_x), dims=(2, 3))
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Эпоха {epoch+1} ")
    
    torch.save(model.state_dict(), MODEL)
else:
    model.load_state_dict(torch.load(MODEL, map_location=device))
    print("Мозги загружены!")

def predict_robust(path):
    if not os.path.exists(path): return print(f"Файл {path} не найден!")
    
    img = Image.open(path).convert('L')
    img = ImageOps.invert(img)
    
    bbox = img.getbbox()
    if bbox: img = img.crop(bbox)
    img = ImageOps.expand(img, border=10, fill=0)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_arr = (np.array(img) / 255.0 - 0.1307) / 0.3081
    img_tensor = torch.FloatTensor(img_arr).reshape(1, 1, 28, 28).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(img_tensor), dim=1)
        pred = torch.argmax(probs).item()
        conf = torch.max(probs).item() * 100

    plt.imshow(img, cmap='gray')
    plt.title(f"ИИ ВИДИТ: {pred} ({conf:.2f}%)")
    plt.axis('off'); plt.show()

predict_robust("num.png")
