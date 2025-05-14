import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.metrics import classification_report
import numpy as np
import random
import logging
from datetime import datetime
import os

# Definir a seed para reproducibilidade
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configurar o logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"log_{current_time}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 30
batch_size = 8
learning_rate = 0.0001
num_workers = min(multiprocessing.cpu_count(), 8)

# Verificar CUDA
logger.info(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memória alocada inicialmente: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

# 1. Definir transformações
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Carregar datasets
full_train_dataset = ImageFolder(root='chest_xray/train', transform=train_transform)
test_dataset = ImageFolder(root='chest_xray/test', transform=test_transform)

# Dividir o conjunto de treino em treino e validação (80% treino, 20% validação)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Verificar classes
logger.info(f"Classes: {full_train_dataset.classes}")
logger.info(f"Tamanho do conjunto de treino: {len(train_dataset)}")
logger.info(f"Tamanho do conjunto de validação: {len(val_dataset)}")

# 3. Definir a arquitetura da CNN
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Congelar camadas iniciais (fine-tuning)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# 4. Definir função de perda e otimizador
num_normal = len([x for x, y in full_train_dataset.imgs if full_train_dataset.classes[y] == 'NORMAL'])
num_pneumonia = len([x for x, y in full_train_dataset.imgs if full_train_dataset.classes[y] == 'PNEUMONIA'])
total = num_normal + num_pneumonia
weight_normal = 1.2 * total / (2 * num_normal)
weight_pneumonia = total / (2 * num_pneumonia)
class_weights = torch.tensor([weight_normal, weight_pneumonia]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# Inicializar o otimizador com os parâmetros que têm requires_grad=True (apenas a camada final inicialmente)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 5. Função para treinar o modelo
def train_model():
    model.train()
    total_step = len(train_loader)
    loss_list = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        if epoch == 5:  # Descongelar todas as camadas após 5 épocas
            for param in model.parameters():
                param.requires_grad = True
            # Atualizar o otimizador para incluir todos os parâmetros
            global optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if (i + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
                logger.info(f"Memória alocada: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
        # Validação
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        logger.info(f'Época [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping na época {epoch+1}")
                break
        model.train()
        scheduler.step(val_loss)
    logger.info('Treinamento concluído!')
    
    plt.plot(loss_list)
    plt.xlabel('Iteração')
    plt.ylabel('Loss')
    plt.title('Perda durante o treinamento')
    plt.show()

# 6. Função para avaliar o modelo
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        logger.info(f'Acurácia no conjunto de teste: {accuracy:.2f}%')
        logger.info("\nRelatório de Classificação:")
        report = classification_report(all_labels, all_preds, target_names=full_train_dataset.classes)
        for line in report.split('\n'):
            logger.info(line)

# 7. Executar treinamento e avaliação
if __name__ == '__main__':
    logger.info("Iniciando teste de treinamento...")
    train_model()
    evaluate_model()

# 8. Salvar o modelo treinado
torch.save(model.state_dict(), 'pneumonia_model.pth')