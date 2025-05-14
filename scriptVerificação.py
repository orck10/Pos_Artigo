import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
import time

# 1. Verificar se CUDA está disponível
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Número de GPUs disponíveis: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Nenhuma GPU CUDA detectada. Usando CPU.")

# 2. Criar um modelo simples
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 3. Criar um dataset simulado
fake_dataset = FakeData(size=1000, image_size=(3, 16, 16), num_classes=2, transform=transforms.ToTensor())
train_loader = DataLoader(fake_dataset, batch_size=32, shuffle=True, num_workers=4)

# 4. Instanciar o modelo e mover para dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# 5. Testar DataParallel (se múltiplas GPUs)
if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel!")
    model = nn.DataParallel(model)
elif torch.cuda.is_available():
    print("Usando 1 GPU.")
else:
    print("Usando CPU.")

# 6. Função para treinar e medir desempenho
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    total_time = 0

    for epoch in range(2):
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/2], Step [{i+1}], Loss: {loss.item():.4f}")
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"Tempo da época {epoch+1}: {epoch_time:.2f} segundos")
    print(f"Tempo total de treinamento: {total_time:.2f} segundos")

# 7. Executar o treinamento
if __name__ == '__main__':
    print("Iniciando teste de treinamento...")
    train_model()
    print("Teste concluído!")