import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import deepfake_detection.utils as ut
from deepfake_detection.data_utils import mesonet_data_transforms
from deepfake_detection.models.MESO4 import Meso4
from deepfake_detection.models.MESO_INCEPTION4 import MesoInception4
def train(model_name, train_path, val_path, save_as, continue_train, trained_model_path, epochs, batch_size):
    torch.backends.cudnn.benchmark=True
    device = ut.get_device()
    trained_model_saving_path = 'deepfake_detection/trained_models/'+ model_name
    if not os.path.exists(trained_model_saving_path):
        os.mkdir(trained_model_saving_path)
    train_data = torchvision.datasets.ImageFolder(train_path, transform = mesonet_data_transforms['train'])
    val_data = torchvision.datasets.ImageFolder(val_path, transform = mesonet_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    
    train_data_size = len(train_data)
    val_data_size = len(val_data)
    
    if model_name.lower() == 'meso4':
        model = Meso4()
    elif model_name.lower() == 'mesoinception4':
        model = MesoInception4()
    if continue_train:
        model.load_state_dict(torch.load(trained_model_path))
    torch.save(model.state_dict(), os.path.join(trained_model_saving_path, model_name + '.pkl'))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_model_weights = model.state_dict()
    best_accuracy = 0
    iteration = 0
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-'*10)
        model = model.train()
        train_loss = 0
        train_corrects = 0
        val_loss = 0
        val_corrects = 0
        for (images, labels) in train_loader:
            iter_loss = 0
            iter_corrects = 0
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration%20):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
        epoch_loss = train_loss / train_data_size
        epoch_acc = train_corrects / train_data_size
        print('Epoch train loss: {:.4f}, accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
        
        model.eval()
        with torch.no_grad():
            for (images, labels) in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = val_loss / val_data_size
            epoch_acc = val_corrects / val_data_size
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = model.state_dict()
                print('Val accuracy got improved. Saving model...')
                torch.save(model.state_dict(), os.path.join(trained_model_saving_path,  str(epoch) + '_' + save_as))
        scheduler.step()
    print('Best val accuracy: {:.4f}. Saving model...'.format(best_accuracy))
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), os.path.join(trained_model_saving_path,  model_name + "_best.pkl"))  
    return model
