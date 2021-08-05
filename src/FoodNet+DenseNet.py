# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:19:45 2021

@authors: Nooreldean Koteb & Abdallah Soliman
"""


import os
import json
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn
from torch.optim import Adam, SGD
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from itertools import product
from sklearn.model_selection import train_test_split
from random import randint
from collections import OrderedDict



    
class foodNet():
    def __init__(self, folder, ):
        print('Creating New FoodNet...')
        self.images = folder
        self.use_gpu()


        self.best_model = None
        self.best_params = None
        self.best_acc = None

    def create(self, n_layer, image_size, n_hidden=16, kernel=5, dropout=None, n_classes=101):
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(512, 101)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        return model

    def use_gpu(self,):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}\n')

        if self.device.type == 'cuda':
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
            print(f'Cached:    {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB\n')
        # else:
        #     torch.set_default_tensor_type(torch.FloatTensor)
    def preprocess(self,batch_size, rotation, resize, final=False, split=[0.7, 0.15, 0.15]):
        print('Preprocessing data...')

        transform = transforms.Compose([
                                        #transforms.Resize(resize),
                                        # transforms.CenterCrop(224),
                                        transforms.RandomResizedCrop(resize),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(rotation),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                       ])

        self.dataset = datasets.ImageFolder(self.images, transform=transform)
        self.labels_map = self.dataset.class_to_idx

        if final == False:
            n = len(self.dataset)
            train, valid, test = torch.utils.data.random_split(self.dataset, [int(n*split[0]), int(n*split[1]), int(n*split[2])])

            self.train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

            self.valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

        else:
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)


    def cross_val(self, parameters):
        self.results = {}

        keys, values = zip(*parameters.items())
        combinations = []
        for val in product(*values):
            combinations.append(dict(zip(keys, val)))

        for param in combinations:
            print(f'\nValidating parameters: {param}')
            self.preprocess(batch_size=param['batch_size'],
                            rotation=param['rotation'],
                            resize=param['resize'],
                            split=param['split'])

            print('Creating a new model...')
            self.model = self.create(n_layer=param['n_layers'],
                             image_size = param['resize'],
                             n_hidden = param['n_hidden'],
                             kernel = param['kernel'],
                             dropout=param['dropout'],
                             n_classes=len(self.labels_map)).to(self.device)

            self.best_model = self.create(n_layer=param['n_layers'],
                             image_size = param['resize'],
                             n_hidden = param['n_hidden'],
                             kernel = param['kernel'],
                             dropout=param['dropout'],
                             n_classes=len(self.labels_map)).to(self.device)

            self.epochs = param['epochs']
            self.crit = nn.CrossEntropyLoss()
            exec(f"self.optimizer = torch.optim.{param['optimizer'][0]}(self.model.parameters(), lr=param['optimizer'][1])")
            self.patience = param['patience']




            print(self.model)
            loss, acc = self.train()

            with open(f'model-{acc}.txt', 'w') as f:
                f.write(str(param))
            f.close()

            self.results[acc] = param
            if self.best_acc == None or self.best_acc < acc:
                print('Best Model Updated!')
                self.best_acc = acc
                self.best_params = param
                self.best_model.load_state_dict(self.model.state_dict())


    def train(self, valid=True):
        print('\nTraining Model...')
        steps = len(self.train_loader)
        t = dt.datetime.now()

        self.validation = []

        for epoch in range(self.epochs):
            if epoch != 0:
                print(f'\repoch: {epoch}/{self.epochs} - 100%: {loss.item() :.4f}\n', end='', flush=True)

            if valid == True and epoch != 0:
                if epoch%self.patience == 0:
                    vald_results = self.valid()

                    print(f'\rValidation Test Accuracy after epoch-{epoch}: {vald_results :.2f}%\n')

                    if self.validation == []:
                        self.validation = [vald_results, self.model.state_dict()]
                    else:
                        if vald_results < self.validation[0]:
                            print('\nNo further Improvement. Taking last best model!')
                            self.model.load_state_dict(self.validation[1])
                            break
                        else:
                            self.validation = [vald_results, self.model.state_dict()]

            for i, (images, labels) in enumerate(self.train_loader):
                outputs = self.model(images.to(self.device))
                loss = self.crit(outputs, labels.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                if (dt.datetime.now() -t).seconds / 10 >= 1:
                    t = dt.datetime.now()
                    percent = ((i/steps)*100)
                    print(f'\repoch: {epoch+1}/{self.epochs} - {percent :.2f}%: {loss.item() :.4f}', end='', flush=True)

        acc = self.test()

        return (loss.item(), acc)


    def test(self):
        print('Testing Model...\n')
        self.model.eval()

        steps = len(self.test_loader)
        t = dt.datetime.now()

        with torch.no_grad():
            right = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                outputs = self.model(images.to(self.device))
                _, pred = torch.max(outputs.data, 1)

                labels = labels.to(self.device)

                total += labels.size(0)
                right += (pred==labels).sum().item()


                if (dt.datetime.now() -t).seconds / 10 >= 1:
                    t = dt.datetime.now()
                    percent = ((i/steps)*100)
                    print(f'\rTest - {percent :.2f}% Done', end='', flush=True)

        percent = ((right / total)*100)
        print(f'\rTest Accuracy: {percent :.2f} %\n', end='', flush=True)
        torch.save(self.model.state_dict(), f'model-{randint(0, 100)}-{percent}.ckpt')

        return percent

    def valid(self):
        print('Validating Model...\n')
        self.model.eval()

        steps = len(self.valid_loader)
        t = dt.datetime.now()

        with torch.no_grad():
            right = 0
            total = 0
            for i, (images, labels) in enumerate(self.valid_loader):
                outputs = self.model(images.to(self.device))
                _, pred = torch.max(outputs.data, 1)

                labels = labels.to(self.device)

                total += labels.size(0)
                right += (pred==labels).sum().item()

                if (dt.datetime.now() -t).seconds / 10 >= 1:
                    t = dt.datetime.now()
                    percent = ((i/steps)*100)
                    print(f'\rValidation - {percent :.2f}% Done', end='', flush=True)
        return ((right / total)*100)


    def pred(self, folder, model=None):
        print('Predicting Results...')
        self.images = folder
        self.preprocess(self.best_params['batch_size'], 
                        self.best_params['rotation'], 
                        self.best_params['resize'], final=True)
        if model != None:
            self.best_model = model
            
        self.best_model.eval()

        steps = len(self.data_loader)
        t = dt.datetime.now()

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_loader):
                outputs = self.best_model(images.to(self.device))
                _, pred = torch.max(outputs.data, 1)

                if (dt.datetime.now() -t).seconds / 10 >= 1:
                    t = dt.datetime.now()
                    percent = ((i/steps)*100)
                    print(f'\rPredicting - {percent :.2f}% Done', end='', flush=True)
            print(f'\rPredicting - 100% Done', end='', flush=True)
        return pred




# # for images, labels in dataloader:
# images, labels = next(iter(dataloader))
# grid = make_grid(images, nrow=5)
# plt.figure(figsize=(15,15))
# plt.imshow(np.transpose(grid, (1,2,0)))
# print('labels:', labels)

params = {
    'n_layers': [2,3],
    'dropout': [None, .1, .25],
    'n_hidden': [8, 16, 32],
    'kernel': [5],

    'split': [[0.7, 0.15, 0.15], ],
    'patience': [5],
    'epochs': [200],
    'optimizer': [('Adam', 0.0001), ('Adam', 0.00001), ('Adam', 0.00001)],#, ('SGD', 0.001)
    'batch_size': [32],
    'rotation': [30],
    'resize': [224],

    }

foodNet = foodNet(folder='images/')
foodNet.cross_val(params)
#results = foodNet.pred(folder='images/predict')



