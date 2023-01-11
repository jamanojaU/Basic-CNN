import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from helper_functions import accuracy_fn
from timeit import default_timer as timer
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from pathlib import Path


class FashionMNISTModel(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(  in_channels=input_features, 
                        out_channels=hidden_units, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(  in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(  in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_features)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x





def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):

  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time


"""
This function is used to train a model.
  Args:
    - model: The model to be trained
    - train_loader: The training data to be used for training
    - loss_fn: The loss function to be used for training
    - optimizer: The optimizer to be used for training
    - accuracy_fn: The accuracy function to be used for training
    - device: The device to be used for training
  Returns:
    - None (prints out the train loss and accuracy)
"""

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
  train_loss, train_acc = 0, 0

  # Put model into training mode
  model.train()

  # Add a loop to loop through the training batches
  for batch, (X, y) in enumerate(data_loader):
    # Put data on target device 
    X, y = X.to(device), y.to(device)

    # 1. Forward pass (outputs the raw logits from the model)
    y_pred = model(X)
    
    # 2. Calculate loss and accuracy (per batch)
    loss = loss_fn(y_pred, y)
    train_loss += loss # accumulate train loss
    train_acc += accuracy_fn(y_true=y,
                             y_pred=y_pred.argmax(dim=1))
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4. Loss backward
    loss.backward()
    
    # 5. Optimizer step (update the model's parameters once *per batch*)
    optimizer.step()
  
  # Divide total train loss and acc by length of train dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


"""
This function is used to test the model on a test set.
  Args:
    - model: The model to be tested
    - data_loader: The data to be used for testing
    - loss_fn: The loss function to be used for testing
    - accuracy_fn: The accuracy function to be used for testing
    - device: The device to be used for testing
  Returns:
    - None (prints out the test loss and accuracy)
  """

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
  
  test_loss, test_acc = 0, 0
  
  # Put the model in eval mode
  model.eval()

  # Turn on inference mode context manager
  with torch.inference_mode():
    for X, y in data_loader:
      # Send the data to the target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass (outputs raw logits)
      test_pred = model(X)

      # 2. Calculuate the loss/acc
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true=y,
                              y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels 

    # Adjust metrics and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")


"""
This function is used to make predictions on a model.
  Args:
    - model: The model to be used for predictions
    - data: The data to be used for predictions
    - device: The device to be used for predictions
  Returns:
    - pred_probs: The prediction probabilities
"""

def make_predictions(model: torch.nn.Module,
                    data: list,
                    device: torch.device):
  
  pred_probs = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample in data:
      #Prepare the sample
      sample = torch.unsqueeze(sample, dim=0).to(device)

      #Forward pass
      pred_logit = model(sample)

      #Get prediction probability
      pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

      pred_probs.append(pred_prob.cpu())

  return torch.stack(pred_probs)



"""
This function is used to plot the confusion matrix.
  Args:
    - model: The model to be used for predictions
    - test_loader: The test loader to be used for predictions
    - device: The device to be used for predictions
  Returns:
    - None (plots the confusion matrix)
"""

def Plot_Confusion_Matrix(model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          pred_probs: torch.Tensor,
                          test_labels: torch.Tensor,
                          device: torch.device):
  pred_classes = pred_probs.argmax(dim = 1)
  print(pred_classes)
  print(test_labels)

  #Let's make a confusion matrix
  y_preds = []
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(test_loader, desc="Making predictions"):
      X, y = X.to(device), y.to(device)
      y_logit= model(X)
      y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

      y_preds.append(y_pred.cpu())

  y_preds = torch.cat(y_preds)

  #Setup the confusion instance and compare results
  cm = ConfusionMatrix(num_classes=10, task="multiclass")
  cm_tensor = cm(preds = y_preds,
                target = test_dataset.targets)

  #Plot the confusion matrix
  fig, ax = plot_confusion_matrix(conf_mat=cm_tensor.numpy(),
                                  figsize=(10,10),
                                  show_normed=True,
                                  colorbar=True)

  plt.show()



"""
This function is used to save the model.
  Args:
    - model: The model to be saved
  Returns:
    - None (saves the model)
"""
def SAVE_MODEL(model: torch.nn.Module):
  MODEL_PATH = Path("models")
  MODEL_PATH.mkdir(parents = True, exist_ok=True)

  MODEL_NAME = "FashionMNISTModel.pth"
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

  print(f"Saving model to {MODEL_SAVE_PATH}...")
  torch.save(model.state_dict(), MODEL_SAVE_PATH)



"""
This function is used to load the model.
  Args:
    - model: The model to be loaded
  Returns:
    - model: The loaded model
"""
def LOAD_MODEL(model: torch.nn.Module):
  MODEL_PATH = Path("models")
  MODEL_NAME = "FashionMNISTModel.pth"
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

  print(f"Loading model from {MODEL_SAVE_PATH}...")
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))

  return model


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTModel(input_features=1, output_features=10, hidden_units=32).to(device)


ttsm = timer()

epochs = 5

#We take our data from MNIST and we transform it to tensors
train_dataset = FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#Define now the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(epochs):
  print(f"Epoch: {epoch}\n-------")
  train_step(model=model,
             data_loader=train_loader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=ttsm,
                                            end=train_time_end_model_2,
                                            device=device)
     
print(f"Total training time: {total_train_time_model_2}")


test_samples =  []
test_labels = []

for samples, label in random.sample(list(test_dataset), k=9):
  test_samples.append(samples)
  test_labels.append(label)

test_samples[0].shape


pred_probs = make_predictions(model=model,
                              data=test_samples,
                              device=device)

#Convert prediction probabilities to labels

pred_classes = pred_probs.argmax(dim = 1)
print(pred_classes)
print(test_labels)

#Let's make a confusion matrix
Plot_Confusion_Matrix(model=model, test_loader=test_loader, pred_probs=pred_probs, test_labels=test_labels, device=device)

