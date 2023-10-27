import torch
from tqdm.auto import tqdm
import numpy as np
from model import *
from datasets import create_datasets, create_data_loaders
from utils import *


plot = True

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Computation device: {device}\n")

# training
def train(model, epoch, trainloader, optimizer, 
          criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    if plot:
        embdeeings, all_labels = [], []

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        # print(type(image), type(labels))
        image = image.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(image)
        # print(type(outputs))
        # print(outputs.size())
        # print(outputs)
        # calculate the loss
        features, loss = criterion(outputs, labels)
        features = F.normalize(features)
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()


        if plot:
            embdeeings.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        train_running_loss += loss.item()
    
    if plot:
        embdeeings = np.concatenate(embdeeings)
        all_labels = np.concatenate(all_labels)
        sphere_plot(embdeeings, all_labels, epoch, figure_path='./plots/train')
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc



# validation
def validate(model, epoch, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    if plot:
        embeddings, all_labels = [], []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            features, loss = criterion(outputs, labels)
            features = F.normalize(features)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        if plot:
            embeddings.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
    
    if plot:
        embeddings = np.concatenate(embeddings)
        all_labels = np.concatenate(all_labels)
        sphere_plot(embeddings, all_labels, epoch, figure_path='./plots/test')
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc