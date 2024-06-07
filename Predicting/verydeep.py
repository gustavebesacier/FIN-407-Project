from lasso_elastic import clean_semicolumns, prepare_data, prepare_lagged_data


import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm
from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

SIZE_FIRST_LINEAR = None

class VeryDeep(nn.Module):
    def __init__(self, size_first_layer):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size_first_layer, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 200),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 140), 
            nn.LeakyReLU(),
            nn.Linear(140, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 200),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(200, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 350),
            nn.LeakyReLU(),
            nn.Linear(350, 200),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(200, 120), 
            nn.LeakyReLU(),
            nn.Linear(120, 50), 
            nn.LeakyReLU(),
            nn.Linear(50, 1)
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


def get_tensors_data(data,):
    
    features, target, pred = prepare_data(data)
    
    # Scale and split data
    X, scaler = standardize_data(features)
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

    # Convert to tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


def get_dataloaders(X_train, X_test, y_train, y_test):

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    return device


def epoch_TRAIN(model, optimizer, criterion, metrics, train_loader, device = get_device(), verbose = False):
    true_values = []
    pred_values = []

    
    EPOCH_LOSS = 0
    METRICS = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    for k, (x, y) in tqdm(enumerate(train_loader)):
        
        model.train()
        
        predictors = x.to(device)
        target_return = y.to(device)

        optimizer.zero_grad() # clear gradient

        predicted_return = model(predictors) # feed the model with predictors

        # print("PREDICTION: ", predicted_return.squeeze())
        # print("TRUE VALUE: ", target_return.squeeze())
        true_values.append(target_return)
        pred_values.append(predicted_return)
        loss = criterion(predicted_return, target_return) # compute the loss between true and predicted values

        loss.backward()     # back pass in the model
        optimizer.step()    # optimize

        # Compute metrics to keep track of the evolution
        with torch.no_grad():
            for metric in METRICS.keys():
                METRICS[metric] += metrics[metric](predicted_return, target_return.squeeze())

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

            loss = criterion(predicted_return, target_return) # compute the loss between true and predicted values

            # Compute metrics to keep track of the evolution
            for metric in METRICS.keys():
                METRICS[metric] += metrics[metric](predicted_return, target_return.squeeze())

            LOSS += loss.item() # sum all losses from batches

    LOSS /= len(test_loader) # average the loss

    for metric in METRICS.keys():
          METRICS[metric] /= len(test_loader) # average the metrics

    clear_output() 

    if verbose:
        print(f"LOSS: {LOSS:.4f} --", ", ".join([f'{k}: {METRICS[k]:.4f}' for k in METRICS.keys()]))
    
    return LOSS,  METRICS

def plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs, epoch, lr, show = False, save = False):
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
        x = str(lr).split(".")[1]
        plt.savefig(f"figures/report_{x}_epoch_{epoch}")
    if show:
        plt.show()

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    """THIS FUNCTION IS NOT OUR WORK AND IS EXTRACTED FROM LABS OF EPFL COURSE 'EE-559 DEEP-LEARNING'
    AND WAS DEVELOPED BY IDIAP RESEARCH LAB"""
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

def train_cycle(model, optimizer, criterion, metrics, train_loader, test_loader, n_epochs, device, lr, show = False, save = False):
    """THIS FUNCTION IS NOT OUR WORK AND IS EXTRACTED FROM LABS OF EPFL COURSE 'EE-559 DEEP-LEARNING'
    AND WAS DEVELOPED BY IDIAP RESEARCH LAB"""
    train_loss_log,  test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]


    for epoch in range(n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs))
        train_loss, train_metrics = epoch_TRAIN(model, optimizer, criterion, metrics, train_loader, device)

        test_loss, test_metrics = epoch_EVAL(model, criterion, metrics, test_loader, device)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(metrics_names, test_metrics_log, test_metrics)

        results_models_weights_dir = 'verydeep_weights/'
        
        if not os.path.exists(results_models_weights_dir):
            os.mkdir(results_models_weights_dir)
        torch.save(model.state_dict(), results_models_weights_dir + 'verydeep.pth')

        plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, test_metrics_log, epoch, lr = lr, show = show, save = save)

    return train_metrics_log, test_metrics_log

def train_model(path_data, epochs):

    data = clean_semicolumns(pd.read_csv(path_data))

    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = get_tensors_data(data)

    # Create the dataloaders
    train_loader, test_loader = get_dataloaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    
    # Create instance of the class for the model
    model = VeryDeep(X_train_tensor.shape[1]) # the parameter is used to specify the size of the input for the 1st layer

    # Print some informations about the model
    print(model)
    get_model_description(model, X_train_tensor[0].shape)

    # Specify the model evaluation metrics
    metrics = {'MAE': mean_absolute_error, "MAPE": mean_absolute_percentage_error}

    # Specify the optimizer and the loss function
    LR = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Run a training epoch
    # EPOCH_LOSS, METRICS = epoch_TRAIN(model, optimizer, criterion, metrics, train_loader, device=get_device(), verbose=True)
    # EPOCH_LOSS, METRICS = epoch_EVAL(model, criterion, metrics, test_loader, device=get_device(), verbose=True)

    train_metrics_log, test_metrics_log = train_cycle(model, 
                                                  optimizer,
                                                  criterion, 
                                                  metrics, 
                                                  train_loader, 
                                                  test_loader, 
                                                  n_epochs= epochs,
                                                  show = False, 
                                                  save = True, 
                                                  device=get_device(),
                                                  lr = LR)
    
    results_models_weights_dir = 'verydeep_weights/'
    if not os.path.exists(results_models_weights_dir):
        os.mkdir(results_models_weights_dir)
    torch.save(model.state_dict(), results_models_weights_dir + 'verydeep.pth')

    return model, scaler


if __name__ == "__main__":

    # data = clean_semicolumns(pd.read_csv("Data/final_data_198501_200001.csv"))
    # # data = prepare_lagged_data(data)

    # # Format the data in tensor form
    # X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, _ = get_tensors_data(data)

    # # Create the dataloaders
    # train_loader, test_loader = get_dataloaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    
    # # Create instance of the class for the model
    # model = VeryDeep(X_train_tensor.shape[1]) # the parameter is used to specify the size of the input for the 1st layer

    # # Print some informations about the model
    # print(model)
    # get_model_description(model, X_train_tensor[0].shape)

    # # Specify the model evaluation metrics
    # metrics = {'MAE': mean_absolute_error, "MAPE": mean_absolute_percentage_error}

    # # Specify the optimizer and the loss function
    # LR = 0.0001
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # criterion = nn.MSELoss()

    # # Run a training epoch
    # # EPOCH_LOSS, METRICS = epoch_TRAIN(model, optimizer, criterion, metrics, train_loader, device=get_device(), verbose=True)
    # # EPOCH_LOSS, METRICS = epoch_EVAL(model, criterion, metrics, test_loader, device=get_device(), verbose=True)

    # train_metrics_log, test_metrics_log = train_cycle(model, 
    #                                               optimizer,
    #                                               criterion, 
    #                                               metrics, 
    #                                               train_loader, 
    #                                               test_loader, 
    #                                               n_epochs= 4,
    #                                               show = False, 
    #                                               save = True, 
    #                                               device=get_device(),
    #                                               lr = LR)
    
    # results_models_weights_dir = 'verydeep_weights/'
    # if not os.path.exists(results_models_weights_dir):
    #     os.mkdir(results_models_weights_dir)
    # torch.save(model.state_dict(), results_models_weights_dir + 'verydeep.pth')

    train_model("../Data/final_data_198501_200001.csv")