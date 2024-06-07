import sys
sys.path.insert(0, '..')

from ModellingOptimization.lasso_elastic import clean_semicolumns, prepare_data

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from tqdm import tqdm
from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

SIZE_FIRST_LINEAR = None

def cohen(preds, target):
    return cohen_kappa_score(target, preds)

def f1(preds, target):
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    return accuracy_score(target, preds)

class WarmupThenCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
        Custom learning rate scheduler that combines warmup and cosine annealing schedules.

        :param optimizer: The optimizer for which to schedule the learning rate.
        :param warmup_scheduler: The scheduler used for the warmup phase.
        :param cosine_scheduler: The scheduler used for the cosine annealing phase.
        :param num_warmup_steps: The number of steps for the warmup phase.
        """
    def __init__(self, optimizer, warmup_scheduler, cosine_scheduler, num_warmup_steps):
        self.warmup_scheduler = warmup_scheduler
        self.cosine_scheduler = cosine_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.step_count = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.step_count < self.num_warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.cosine_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.step_count < self.num_warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step(epoch)
        self.step_count += 1

class Flash(nn.Module):
    def __init__(self, size_first_layer):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size_first_layer, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


def count_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_description(model:nn.Module, tensor_shape):
    summary(model, tensor_shape)
    print("Number of parameters: ", count_parameters(model))


def standardize_data(X_train_raw):
    """Takes the raw predictors for training and testing and return them after standardization and also returns the scaler"""
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)

    return X_train, scaler


def get_tensors_data(data, eval=False):
    
    features, target, _ = prepare_data(data, order_data=False, dep_variable="trading_signal")
    
    # Scale and split data
    X, scaler = standardize_data(features)

    if eval:
        X_eval_tensor = torch.tensor(X, dtype=torch.float32)
        y_eval_tensor =  torch.tensor(target, dtype=torch.float32)
        return X_eval_tensor, y_eval_tensor
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

        # Convert to tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


def get_dataloaders(X_train, X_test, y_train, y_test):

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    return device


def epoch_TRAIN(model, optimizer, criterion, metrics, train_loader, lr_scheduler, device = get_device(), verbose = False):
    true_values = []
    pred_values = []

    EPOCH_LOSS = 0
    METRICS = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    for k, (x, y) in tqdm(enumerate(train_loader)):
        
        model.train()
        
        predictors = x.to(device)
        target_return = y.to(device)

        optimizer.zero_grad() # clear gradient

        output = model(predictors) # feed the model with predictors
        pred = (output > 0.5).float()
        # with torch.no_grad():
        #     _, pred = torch.max(output, -1)

        # print("PREDICTION: ", predicted_return.squeeze())
        # print("TRUE VALUE: ", target_return.squeeze())
        true_values.append(target_return)
        pred_values.append(pred)

        loss = criterion(output, target_return) # compute the loss between true and predicted values

        loss.backward()     # back pass in the model
        optimizer.step()    # optimize
        lr_scheduler.step()

        # Compute metrics to keep track of the evolution
        with torch.no_grad():
            for metric in METRICS.keys():
                METRICS[metric] += metrics[metric](pred, target_return.squeeze())

        EPOCH_LOSS += loss.item() # sum all losses from batches

    EPOCH_LOSS /= len(train_loader) # average the loss

    for metric in METRICS.keys():
          METRICS[metric] /= len(train_loader) # average the metrics

    clear_output() 

    if verbose:
        print(f"LOSS: {EPOCH_LOSS:.4f} --", ", ".join([f'{k}: {METRICS[k]:.4f}' for k in METRICS.keys()]))
    
    return EPOCH_LOSS,  METRICS

def epoch_EVAL(model, criterion, metrics, test_loader, device = get_device(), verbose = False):
    
    model.eval()
    
    LOSS = 0
    METRICS = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    for _, (x, y) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            predictors = x.to(device)
            target_return = y.to(device)

            predicted_return = model(predictors) # feed the model with predictors
            pred = (predicted_return > 0.5).float()
            loss = criterion(predicted_return, target_return) # compute the loss between true and predicted values

            # Compute metrics to keep track of the evolution
            for metric in METRICS.keys():
                METRICS[metric] += metrics[metric](pred, target_return.squeeze())

            LOSS += loss.item() # sum all losses from batches

    LOSS /= len(test_loader) # average the loss

    for metric in METRICS.keys():
          METRICS[metric] /= len(test_loader) # average the metrics

    clear_output() 

    if verbose:
        print(f"LOSS: {LOSS:.4f} --", ", ".join([f'{k}: {METRICS[k]:.4f}' for k in METRICS.keys()]))
    
    return LOSS,  METRICS

def plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs, epoch, date, show = False, save = False):
    """THIS FUNCTION IS NOT OUR WORK AND IS EXTRACTED FROM LABS OF EPFL COURSE 'EE-559 DEEP-LEARNING'
    AND WAS DEVELOPED BY IDIAP RESEARCH LAB"""
    _, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

    ax[0].plot(train_loss, c='blue', label='train')
    ax[0].plot(test_loss, c='orange', label='test')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    for i in range(len(metrics_names)):
        ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
        ax[i + 1].plot(test_metrics_logs[i], c='orange', label='test')
        ax[i + 1].set_title(metrics_names[i])
        ax[i + 1].set_xlabel('epoch')
        ax[i + 1].legend()

    if save:
        # x = str(lr).split(".")[1]
        plt.savefig(f"figures/{date}_report_epoch_{epoch}")
    if show:
        plt.show()

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    """THIS FUNCTION IS NOT OUR WORK AND IS EXTRACTED FROM LABS OF EPFL COURSE 'EE-559 DEEP-LEARNING'
    AND WAS DEVELOPED BY IDIAP RESEARCH LAB"""
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

def train_cycle(model, optimizer, criterion, metrics, train_loader, test_loader, n_epochs, device, date, lr_scheduler, show = False, save = False):
    """THIS FUNCTION IS NOT OUR WORK AND IS EXTRACTED FROM LABS OF EPFL COURSE 'EE-559 DEEP-LEARNING'
    AND WAS DEVELOPED BY IDIAP RESEARCH LAB"""

    train_loss_log,  test_loss_log = [], []
    
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for _ in range(len(metrics))]
    test_metrics_log = [[] for _ in range(len(metrics))]


    for epoch in range(n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs))
        train_loss, train_metrics = epoch_TRAIN(model, optimizer, criterion, metrics, train_loader, lr_scheduler, device, verbose = True)

        test_loss, test_metrics = epoch_EVAL(model, criterion, metrics, test_loader, device, verbose = True)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(metrics_names, test_metrics_log, test_metrics)

        results_models_weights_dir = 'FLASH_weights/'
        if not os.path.exists(results_models_weights_dir):
            os.mkdir(results_models_weights_dir)
        torch.save(model.state_dict(), results_models_weights_dir + f'FLASH_202206.pth')

        plot_training(
            train_loss = train_loss_log, 
            test_loss = test_loss_log, 
            metrics_names = metrics_names, 
            train_metrics_logs = train_metrics_log, 
            test_metrics_logs = test_metrics_log, 
            epoch = epoch, 
            date = date, 
            show = show, 
            save = save
            )

        inference(path_data="Data/Test_202001_202206.csv", show = True)

    return train_metrics_log, test_metrics_log

def inference(path_data, show = True):

    # Load the data
    data = clean_semicolumns(pd.read_csv(path_data))

    # Add a trading signal column
    data['trading_signal'] = data['return_lag'].transform(lambda x: 1 if x > 0 else 0)
    data.drop('return_lag', axis=1, inplace=True)
    
    # Get the date from the name of the file to the load the correct weights
    # date = "_".join(path_data.split(".")[0].split("/")[1].split("_")[2:4])
    date = "_".join(path_data.split(".")[0].split("/")[1].split("_")[2:4])

    # Split the data
    X_eval, y_eval = get_tensors_data(data, eval = True)

    # Load the model and the weights
    model = Flash(X_eval.shape[1])
    model.load_state_dict(torch.load(f"FLASH_weights/FLASH_{date}.pth"))
    model.eval()

    # Get predictions
    pred = (model(X_eval) > 0.5).float()
    data['pred_trading_signal'] = pred.squeeze().numpy() # Add prediction column to the dataframe

    data.to_csv("DATA_AVEC_LES_PREDICTIONS.csv")

    # Get metrics
    accuracy_eval = acc(pred, y_eval)
    cohen_eval = cohen(pred, y_eval)

    if show:
        print(f"Evaluation metrics for period {date}:", 
              f" - Accuracy: {round(accuracy_eval, 4)}",
              f" - Cohen      : {round(cohen_eval, 4)}",
              sep = "\n")


def train_model(path_data, epochs):

    data = clean_semicolumns(pd.read_csv(path_data))
    col = [column for column in data.columns if column not in ['yyyymm', 'permno']]
    data = data[col]
    # print("---------------------------------------------------------")
    
    # Get the date from the name of the file, for saving the weights of the models
    date = "_".join(path_data.split(".")[0].split("/")[1].split("_")[2:4])
    
    # Add a trading signal column
    data['trading_signal'] = data['return_lag'].transform(lambda x: 1 if x > 0 else 0)
    data.drop('return_lag', axis = 1, inplace = True)


    # Split the dataset in eval and train sets
    #data_train, data_eval = train_test_split(data, test_size=0.2, random_state=42)

    # Save the evaluation set in a separate file
    #data_eval.to_csv(path_data.split(".")[0] + "_eval_set.csv")

    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = get_tensors_data(data)

    # Create the dataloaders
    train_loader, test_loader = get_dataloaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    
    # Create instance of the class for the model
    model = Flash(X_train_tensor.shape[1]) # the parameter is used to specify the size of the input for the 1st layer

    # Print some informations about the model
    # print(model)
    # get_model_description(model, X_train_tensor[0].shape)

    # Specify the model evaluation metrics
    metrics = {'Accuracy': acc, "Cohen": cohen}

    # Specify the optimizer and the loss function
    LR = 0.001
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    num_training_steps = epochs * len(train_loader)
    T_0 = 1     # Number of epochs for the first restart
    T_mult = 2  # Increase in the cycles
    num_warmup_steps = 50

    # Create the linear warmup scheduler
    warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                       num_training_steps=num_training_steps)

    # Create the cosine annealing with warm restarts scheduler
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=5e-10)

    # Combine the warmup and cosine annealing schedulers
    lr_scheduler = WarmupThenCosineScheduler(optimizer, warmup_scheduler, cosine_scheduler, num_warmup_steps)

    train_metrics_log, test_metrics_log = train_cycle(
        model, 
        optimizer,
        criterion, 
        metrics, 
        train_loader, 
        test_loader, 
        n_epochs= epochs,
        show = False, 
        save = True, 
        device = get_device(),
        # lr = LR,
        lr_scheduler=lr_scheduler,
        date = date
    )
    

    return model, scaler