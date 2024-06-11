import argparse
from time import time
import torch
import torch.nn as nn
from typing import List
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import preprocess

# Accuracies:

# Data
sequence_length = 1
shuffle = True

# Training
num_epochs = 2
batch_size = 500
learning_rate = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam


# Define Neural Network
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):#, dropout_rate=0.3):
        super(FFNN, self).__init__()
        # Hidden Layers
        self.hidden1 = nn.Linear(input_size, hidden_size)
        #self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        #self.dropout2 = nn.Dropout(dropout_rate)
        # Activation function
        self.relu = nn.ReLU()
        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.relu(out)
        #out = self.dropout1(out)
        out = self.hidden2(out)
        out = self.relu(out)
        #out = self.dropout2(out)
        out = self.output(out)
        return out


# Training function
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Adam,
        num_epochs: int,
        device: torch.device,
) -> List[float]:
    start_time = time()
    print(f"Training Started:")

    model.to(device)
    batch_size = len(train_loader)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            X = X.float().to(device)
            y = y.long().to(device)
            preds = model(X)
            #print(X.shape)
            #print(y.shape, preds.shape)
            preds = preds.squeeze(1)
            loss = loss_fn(preds, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch + 1}/{num_epochs}], Batch: [{i}/{batch_size}], Loss: {loss.item():.4f}, Time: {time() - start_time:.2f}"
                )

    print(f"Total Training Time: {time() - start_time}, Last Loss: {loss.item():.4f}")
    return train_losses


def test_model(test_set, model, loss_fn, device):
    model.eval()
    size = len(test_set)
    num_batches = len(test_set)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i, (X, y) in enumerate(test_set):
            X = X.float().to(device)
            y = y.long().to(device)

            pred = model(X)
            pred = pred.squeeze(1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    accuracy = accuracy * 100
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # retrieve data
    X_train, y_train, X_dev, y_dev, X_test, y_test = preprocess()
    model = FFNN(
        input_size=FEATURE_SIZE,
        hidden_size=64,
        num_classes=CLASS_SIZE,
    )
    optimizer = optimizer_function(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_model(
        model,
        X_test,
        loss_function,
        optimizer,
        num_epochs=num_epochs,
        device=DEVICE,
    )
    # save model
    torch.save(model, f"checkpoints/{args.train_set}_model.pth")

    # test model
    test_model(test_dataloader, model, loss_function, DEVICE)

    # do the same without protected attribute
    # and for predicting protected attribute


if __name__ == "__main__":
    main()

    '''
    
    '''