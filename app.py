import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 10)
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = torch.relu(self.fc1(lstm_out[:, -1, :]))  # Only take last output
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs, use_optimizer):
    model.train()
    train_accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            if use_optimizer:
                optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if use_optimizer:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y_batch, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        train_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy:.2f}%")
    return train_accuracies

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y_batch, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# UI Layout
st.title("LSTM Model Training UI with PyTorch")

# 1) Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("First few rows of the dataset:")
    st.dataframe(data.head())

    # 2) Select Columns for Text (features) and Label (target)
    columns = data.columns.tolist()
    X_columns = st.multiselect("Select feature columns", columns)
    y_column = st.selectbox("Select label column", columns)

    if X_columns and y_column:
        X = data[X_columns].values
        y = data[y_column].values

        # Preprocess Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # Reshape for LSTM
        y_encoded = torch.tensor(pd.get_dummies(y).values).float()  # One-hot encode target

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train).float()
        X_test_tensor = torch.tensor(X_test).float()
        y_train_tensor = torch.tensor(y_train).float()
        y_test_tensor = torch.tensor(y_test).float()

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 3) Set Model Training Parameters
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
        batch_size = st.slider("Batch Size", 16, 128, 32, step=16)
        epochs = st.slider("Epochs", 1, 100, 20, step=1)

        # 4) Select Optimization Method
        optimizer_choice = st.selectbox("Select Optimizer", ["none", "adam", "sgd", "rmsprop", "adagrad", "adamw"])

        # 5) Initialize PyTorch Model and Optimizer
        input_size = X_train_tensor.shape[2]
        output_size = y_train_tensor.shape[1]
        hidden_size = 50  # Example hidden layer size

        model = LSTMModel(input_size, hidden_size, output_size)

        optimizer = None  # Default to None
        use_optimizer = True  # Flag for optimizer usage
        if optimizer_choice == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_choice == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_choice == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_choice == "adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        elif optimizer_choice == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_choice == "none":
            use_optimizer = False  # Disable optimizer usage

        criterion = nn.CrossEntropyLoss()

        # 6) Train Model Button
        if st.button("Start Training"):
            with st.spinner("Training the model..."):
                train_accuracies = train_model(model, train_loader, criterion, optimizer, epochs, use_optimizer)

            # 7) Display Output Metrics
            accuracy = evaluate_model(model, test_loader, criterion)
            st.write(f"Test Accuracy: {accuracy:.2f}%")

            # Accuracy Graph
            st.write("### Training Accuracy Graph")
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epochs + 1), train_accuracies, marker="o")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy (%)")
            plt.title("Training Accuracy over Epochs")
            st.pyplot(plt)

            # 8) Option to Download Model
            if st.button("Download Model"):
                torch.save(model.state_dict(), "lstm_model.pth")
                st.write("Model saved successfully!")

        
