import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt

from ts_nets.utils.early_stopping import EarlyStopping
from ts_nets.graphs.loader import GraphDataset

class BaseModel(nn.Module):
    
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.history = {'train_loss':[], 'val_loss' : []}

        self.args = args
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        

    def fit(self, log:bool = False, verbose:bool = True):
        
        train_loader = DataLoader(GraphDataset(args = self.args, flag='train'),
                                  batch_size= self.args.batch_size, 
                                  shuffle= True, 
                                  num_workers= self.args.num_workers, 
                                  prefetch_factor= self.args.prefetch_factor, 
                                  persistent_workers= self.args.persistent_workers, 
                                  pin_memory= self.args.pin_memory, 
                                  drop_last= self.args.drop_last)
        
        early_stopping = EarlyStopping(patience=self.patience)
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n_batchs = len(train_loader)
        n_samples = (n_batchs * train_loader.batch_size) if train_loader.drop_last else len(train_loader.dataset)
        message = f'Fit in progress ...'
        
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                steps_per_epoch=n_batchs, epochs=self.epochs)
        
        # Training loop
        for epoch in range(self.epochs):
            self.train()  
            train_loss = 0.0
            # Training loop
            for i, (inputs, targets) in enumerate(train_loader):
                if verbose :
                    print(message + f" | Epoch {epoch+1}/{self.epochs} : batch {i+1}/{n_batchs}" + 20*' ', end='\r')
                inputs, targets = inputs.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)
                # Zero the gradients
                optimizer.zero_grad()
                # Autocasting (mixed precision)
                with autocast():
                    # Forward pass
                    outputs = self(inputs)
                    # Compute the loss
                    loss = criterion(outputs, targets)
                # Backward pass and optimization step through scaler object
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item() 

            train_loss = train_loss / n_samples
            
            # Validation step
            val_loss = self._evaluate(criterion, flag = 'val')
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)               
                        
            # Early stopping logic
            message = early_stopping(val_loss, model=self, epoch=epoch, n_epochs=self.epochs, train_loss = train_loss)
            if early_stopping.early_stop:
                if verbose :
                    print('\n' +message, end = '\n')
                # Recording fit arguments 
                self.n_epochs = early_stopping.best_epoch +1
                self.val_loss = - early_stopping.best_score
                self.train_loss = early_stopping.train_loss_best_epoch
                if log :
                    self._log()
                break
            
            # Stopping logic 
            if epoch +1 == self.epochs :
                self.load_state_dict(early_stopping.best_model_weights)
                if verbose : 
                    print('\n' +message, end = '\n')
                    if -early_stopping.best_score < 0.001 :
                        print(f"Restoring best model weights from epoch {early_stopping.best_epoch} with Validation Loss = {-early_stopping.best_score:.4e}")
                    else :
                        print(f"Restoring best model weights from epoch {early_stopping.best_epoch} with Validation Loss = {-early_stopping.best_score:.4f}")               
                if log :
                    self._log()
    
    def _evaluate(self, criterion, flag='val'):
        loader = DataLoader(GraphDataset(args = self.args, flag=flag),
                                  batch_size= self.args.batch_size, 
                                  shuffle= True, 
                                  num_workers= self.args.num_workers, 
                                  prefetch_factor= self.args.prefetch_factor, 
                                  persistent_workers= self.args.persistent_workers, 
                                  pin_memory= self.args.pin_memory, 
                                  drop_last= self.args.drop_last)        

        loss = 0.0
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)
                with autocast():
                    # Forward pass
                    outputs = self(inputs)
                    loss += criterion(outputs, targets).item()
            loss = loss / len(loader.dataset)
        return loss
    
    def predict(self, X_test):        
        self.eval() 
        
        # If X_test is a numpy array, convert it to a PyTorch tensor
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device)
                outputs = self(batch)
                predictions.append(outputs.cpu())
    
        return torch.cat(predictions).numpy()

    def plot_history(self, ax = None):
        if ax is None :
            fig, ax = plt.subplots(1,1,figsize = (6,4))
        ax.plot(np.sqrt(self.history['train_loss']))
        ax.plot(np.sqrt(self.history['val_loss']))
        ax.set_title('Model MSE')
        ax.set_ylabel('MSE')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Val'], loc='best')
        
    def _log(self):
        pass
    
    def _save(self):
        pass        
