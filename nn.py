from time import time
import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.optim as optim
from data_preprocessing import preprocess

# Define device, use CUDA when available:
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Data
sequence_length = 1
shuffle = True

# Training
num_epochs = 10
batch_size = 500
learning_rate = 0.01
loss_function = nn.BCEWithLogitsLoss()
optimizer_function = torch.optim.Adam

def initialize(X_train, X_p_train, y_train,X_dev ,X_p_dev, y_dev, X_test,X_p_test, y_test, _):

    # add protected attribute  
    # X_p_train = X_p_train.reshape(-1,1)
    # X_train = np.concatenate((X_train,X_p_train), axis=1)

    # X_p_dev = X_p_dev.reshape(-1,1)
    # X_dev = np.concatenate((X_dev,X_p_dev), axis=1)

    # X_p_test = X_p_test.reshape(-1,1)
    # X_test = np.concatenate((X_test,X_p_test), axis=1)

    # convert to Torch Tensor
    X_train = torch.from_numpy(X_train)
    X_dev = torch.from_numpy(X_dev)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_dev = torch.from_numpy(y_dev)
    y_test = torch.from_numpy(y_test)
    
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
        train_set: np.ndarray,
        labels: np.ndarray,
        loss_fn: nn.Module,
        optimizer: optim.Adam,
        num_epochs: int,
        device: torch.device,
) -> List[float]:
    start_time = time()
    print(f"Training Started:")

    model.to(device)
    batch_size = len(train_set)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, (X, y) in enumerate(zip(train_set, labels)):
            X = X.float().to(device)
            y = y.float().to(device)
            preds = model(X).squeeze()
            #print(X.shape)
            # print(y.shape)
            # print(y)
            # print(preds.shape)
            # print(preds)
            #preds = preds.unsqueeze(1)
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


def test_model(test_set, test_label, model, loss_fn, device):
    model.eval()
    size = len(test_set)
    num_batches = len(test_set)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i, (X, y) in enumerate(zip(test_set, test_label)):
            X = X.float().to(device)
            y = y.float().to(device)

            pred = model(X).squeeze()
            #pred = pred.squeeze(1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    accuracy = accuracy * 100
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")



def main():
    #retrieve data
    X_train, X_p_train, y_train, X_dev, X_p_dev, y_dev, X_test, X_p_test, y_test, _ = preprocess()
    initialize(X_train, X_p_train, y_train, X_dev, X_p_dev, y_dev, X_test, X_p_test, y_test, _)
    # define and train model
    model = FFNN(
        input_size=X_train.shape[1],
        hidden_size=64,
        num_classes=1,
    )
    optimizer = optimizer_function(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_model(
        model,
        X_train,
        y_train,
        loss_function,
        optimizer,
        num_epochs=num_epochs,
        device=DEVICE,
    )

    # save model
    torch.save(model, "checkpoints/simpleNNmodel.pth")

    # load model
    model = torch.load("checkpoints/simpleNNmodel.pth")

    # test model
    test_model(X_test, y_test, model, loss_function, DEVICE)


if __name__ == "__main__":
    main()

    '''
    
    '''