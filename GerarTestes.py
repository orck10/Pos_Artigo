import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.metrics import classification_report
import numpy as np
import random
from datetime import datetime
from pathlib import Path
import csv
import os
import random



# Definir a seed para Reprodutibilidade
def configs(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
""""

parametros = [
    [6, 6, 0.001, 10, 0.8, 18],
    [7, 8, 0.001, 10, 0.7, 34],
    [6, 10, 0.001, 10, 0.8, 50],
    [8, 10, 0.001, 6, 0.8, 18],
    [6, 10, 0.001, 8, 0.7, 34],
    [7, 10, 0.001, 5, 0.6, 50],
    [6, 10, 0.0001, 10, 0.8, 18],
    [8, 10, 0.00001, 10, 0.8, 34],
    [6, 10, 0.00001, 10, 0.9, 50],
    [7, 10, 0.00001, 10, 0.8, 18],
    [10, 10, 0.001, 10, 0.5, 34],
    [15, 10, 0.001, 10, 0.8, 50],
    [20, 10, 0.001, 10, 0.8, 18],
    [20, 8, 0.0001, 8, 0.6, 34],
    [20, 8, 0.001, 8, 0.7, 50],
    [20, 10, 0.00001, 10, 0.8, 50],
]
"""

def geraParametros():
    par = []
    for i in range (1, 5):
        epocas = random.randint(6,20)
        tamnho_bach = random.randint(5,10)
        learning_rate = round(10 ** random.uniform(-5, -2), 5)
        num_workers = random.randint(3,10)
        tamanho_dataset_treino = round(np.random.choice(np.arange(0.5, 0.801, 0.005)), 2)
        lista_profundidade = [18, 34, 50]
        profundidade = random.choice(lista_profundidade)
        par.append([epocas, tamnho_bach, learning_rate, num_workers, tamanho_dataset_treino, profundidade])

    return par


# 5. Função para treinar o modelo
def train_model(model, train_loader, device, optimizer, scheduler, num_epochs, learning_rate):
    model.train()
    loss_list = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        if epoch == 5:  # Descongelar todas as camadas após 4 épocas
            for param in model.parameters():
                param.requires_grad = True
            # Atualizar o otimizador para incluir todos os parâmetros
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        model.train()
        scheduler.step(val_loss)
    print('Treinamento concluído!')
    
    retorno = [val_accuracy, best_val_loss]
    return retorno


def evaluate_model(model, device, full_train_dataset, test_loader):
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
        report = classification_report(all_labels, all_preds, target_names=full_train_dataset.classes)
        retorno = [accuracy, report]
        return retorno



# Executar treinamento e avaliação
if __name__ == '__main__':

    arquivo = 'arquivo.csv'
    existe = os.path.isfile(arquivo)
    with open(arquivo, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not existe:
            writer.writerow(['Epocas', 'Tamanho do bacth', 'taxa de aprendizagem', 'Numero de workers', 'Tamnho do Data ser de treino %', 'Camadas da Rede', 
                             'Tempo de treino', 'Acuracy do treino', 'Melhor Loss', 'Acuracy real', 'report'])  # header


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

    #Carregar datasets
    full_train_dataset = ImageFolder(root='chest_xray/train', transform=train_transform)
    test_dataset = ImageFolder(root='chest_xray/test', transform=test_transform)


    

    for i in range(1,16):
    #for i in range(1):
        # Definir a seed para Reprodutibilidade

        parametros = geraParametros()
        configs(i)
        print(f"Seed {i} em execução")
        for par in parametros:
            print(f"Parametros - epocas:{par[0]} batch-size:{par[1]} learning-rate:{par[2]} workers:{par[3]} tamanho-dataset-treino:{par[4]}")
            # Configurações

            num_epochs = par[0]
            batch_size = par[1]
            learning_rate = par[2]
            num_workers = min(multiprocessing.cpu_count(), par[3])
            tamanho_data_set_treirno = par[4]

            #Definir a arquitetura da CNN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = None
            if(par[5] == 34):
                model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            elif(par[5] == 50):
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model = model.to(device)


            print(f"CUDA disponível: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
                print(f"Memória alocada inicialmente: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

            # Dividir o conjunto de treino em treino e validação (80% treino, 20% validação)
            train_size = int(tamanho_data_set_treirno * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # Definir função de perda e otimizador
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

            # Congelar camadas iniciais (fine-tuning)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

            incio_treino = datetime.now()
            resultado_treino = train_model(model, train_loader, device, optimizer, scheduler, num_epochs, learning_rate)
            fim_treino = datetime.now()
            resustado_teste = evaluate_model(model, device, full_train_dataset, test_loader)
            
            tempos = [(fim_treino - incio_treino)]
            resultados = par+tempos+resultado_treino+resustado_teste
            with open(arquivo, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(resultados)
                
            print(print(f"Fim Parametros - epocas:{par[0]} batch-size:{par[1]} learning-rate:{par[2]} workers:{{par[3]}} tamanho-dataset-treino:{{par[4]}}"))

        print(f"Fim Seed {i} em execução")

