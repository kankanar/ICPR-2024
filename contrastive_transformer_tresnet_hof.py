# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:48:30 2023

@author: kanroy
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch_geometric.nn import GCNConv
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import time
import os
import copy
from sklearn.metrics import confusion_matrix, classification_report
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import timm
import logging



data_dir = "Hand over face/rgb"
# Number of classes in the dataset
num_classes = 30

# Batch size for training (change depending on how much memory you have)
batch_size = 30

# Number of epochs to train for
num_epochs = 30
def build_edge_idx(num_nodes):
    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes)), dtype=torch.long)
    
    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            E[0, node * (num_nodes) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])
    
    return E
    
class SupConTresNetGCN(nn.Module):
    def __init__(self, name='tresnet', head='mlp', feat_dim = 1024,num_classes = num_classes,                              use_pretrained = True):
        super(SupConTresNetGCN, self).__init__()
        model_tresnet = timm.create_model('tresnet_l', pretrained=use_pretrained, num_classes =                            num_classes)
        #print(torch.load('results and models/Tresnet_l_HOF.pth'))
        #model_tresnet.load_state_dict(torch.load('results and models/Tresnet_l_HOF.pth'))
        ########## Removing last couple of layer sto match input of Vit ###################
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=2432, 
                nhead=4, 
                dim_feedforward=128, 
                dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier =  nn.Sequential(nn.AvgPool2d(12),
                                         nn.Flatten(),
                                         nn.Linear(2432, num_classes))
        
        model_tresnet = nn.Sequential(*list(model_tresnet.children())[:-1])
        self.encoder = model_tresnet
        self.head =  nn.Sequential(nn.AvgPool2d(12),
                                   nn.Conv2d(2432, 1024, 1, stride=1))
        
    def forward(self, x):
            #print(self.edge_index.size())
            b,_,_,_ = x.size()
            aux_x = x[0:b//2,:,:,:]
            #print(aux_x.size())
            feat = self.encoder(x)
            aux_feat  = self.encoder(aux_x)
            #print(aux_feat.size())
            aux_feat = aux_feat.view(b//2,-1,2432)
            aux_feat = self.transformer_encoder(aux_feat)
            aux_feat = aux_feat.view(b//2,2432,12,12)
            #print(aux_feat.size())
            feat_main = self.head(feat)
            feat_main = feat_main.squeeze()
            pred = self.classifier(aux_feat)
            #print(pred.size())
            return feat_main, pred
            
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        #self.csvname = csvname
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
    def forward(self, con_output, con_target, image_output, image_target,phase):
        if phase == 'train':
            precision0 = torch.exp(-self.log_vars[0])
            contrastive_loss = nn.CrossEntropyLoss()(con_output, con_target)
            loss0 = precision0*contrastive_loss + self.log_vars[0]
            image_loss = nn.CrossEntropyLoss()(image_output, image_target)
            precision1 = torch.exp(-self.log_vars[1])
            loss1 = precision1*image_loss + self.log_vars[1]
            return loss0 + loss1
        else:
            image_loss = nn.CrossEntropyLoss()(image_output, image_target)
            return image_loss
def info_nce_loss(features):

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    
    

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
   

    val_acc_history = []
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    n_iter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for images, labels_ in dataloaders[phase]:
                #print(len(images))
                images = torch.cat(images, dim=0)

                images = images.to(device)
                labels_ = labels_.to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with autocast(enabled=fp16_precision):
                    features, predictions = model(images)
                    #print(predictions.size())
                    logits, labels = info_nce_loss(features)
                    loss = criterion(logits, labels, predictions, labels_,phase)
                        

                    # zero the parameter gradients
                    _, preds = torch.max(predictions, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #loss.backward()
                        scaler.scale(loss).backward()

                        scaler.step(optimizer)
                        scaler.update()
                        
                if n_iter % log_every_n_steps == 0:
                    top1, top5 = accuracy(predictions, labels_, topk=(1, 5))
                    writer.add_scalar('loss', loss, global_step=n_iter)
                    writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    #writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
                # statistics
                running_loss += loss.item() * (images.size(0)//2)
                running_corrects += torch.sum(preds == labels_.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # warmup for the first 10 epochs
    

    logging.info("Training has finished.")
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
            

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 384
    model_ft = SupConTresNetGCN(num_classes = num_classes)
    
    return model_ft, input_size

def run():
    torch.multiprocessing.freeze_support()
    
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                     
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight    

if __name__ == '__main__':
    run()
# Initialize the model for this run
    model_ft, input_size = initialize_model(num_classes, use_pretrained=True)
    
    # Print the model we just instantiated
    print(model_ft)
    fp16_precision = True
    scaler = GradScaler(enabled=fp16_precision)
    log_every_n_steps = 10
    n_views = 2
    temperature = 0.07
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': TwoCropTransform(transforms.Compose([
        transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
       'val': TwoCropTransform(transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
       
       'test': TwoCropTransform(transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
    }
    
    print("Initializing Datasets and Dataloaders...")
    dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'))                                                                         
                                                                                
    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    trainsampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
    
    dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val'))                                                                         
                                                                                
    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(dataset_val.imgs, len(dataset_val.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    valsampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
    sampler = {'train':trainsampler,'val':valsampler}
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4,sampler = sampler[x], drop_last= True) for x in ['train', 'val']}
    
    test_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['test']) for x in ['val','test']}
    # Create training and validation dataloaders
    testdataloaders_dict = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=1, shuffle=False, num_workers=4) for x in ['val','test']}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Send the model to GPU
    model_ft = model_ft.to(device)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Setup the loss fxn
    criterion = MultiTaskLossWrapper(2)
    
    
    # Train and evaluate
    #model_ft.load_state_dict(torch.load('vgg16_FER.pth'))
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    torch.save(model_ft.state_dict(), 'tresnet_contrastive+transformer_HOF.pth')
    
    model_ft.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for phase in ['val','test']:
                model_ft.eval()   # Set model to evaluate mode

                # Iterate over data.
                all_preds = torch.tensor([])
                all_labels = torch.tensor([])
                for inputs, labels in testdataloaders_dict[phase]:
                    inputs = torch.cat(inputs, dim=0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    _,outputs = model_ft(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds = torch.cat((all_preds, outputs.argmax(dim=1).detach().cpu()),dim=0)
                    all_labels = torch.cat((all_labels, labels.detach().cpu()),dim=0)
        
                print('Accuracy of the network on the '+ phase+' images: %f %%' % ( 100 * correct / total))  
                conf_mat = confusion_matrix(all_labels, all_preds)
                np.set_printoptions(threshold=np.inf)
                print(conf_mat)
                print(classification_report(all_labels, all_preds))
