import streamlit as st
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.optim import AdamW, SGD, Adam, Adagrad
import shutil
from imblearn.over_sampling import SMOTE
from transformers.utils import logging

logging.set_verbosity_error()

# Function to train the DistilBERT model
def train_distilbert_model(train_texts, train_labels, num_epochs, optimizer_choice,
                           learning_rate, batch_size, warmup_steps, weight_decay,
                           adam_epsilon, max_grad_norm, save_steps, logging_steps,
                           seed, fp16, model_name="distilbert-base-uncased", smote_technique=None):
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name,
                                                                num_labels=len(label_encoder.classes_))

    encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_encodings, val_encodings, train_labels_split, val_labels_split = train_test_split(
        encodings["input_ids"], train_labels, test_size=0.2, random_state=seed
    )

    # SMOTE Resampling (if applicable)
    if smote_technique:
        smote = SMOTE(sampling_strategy='auto', random_state=seed)
        if smote_technique == 'SMOTE':
            train_encodings, train_labels_split = smote.fit_resample(train_encodings, train_labels_split)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset({'input_ids': train_encodings}, train_labels_split)
    val_dataset = TextDataset({'input_ids': val_encodings}, val_labels_split)

    st.write(f"Training dataset size: {len(train_dataset)}")
    st.write(f"Validation dataset size: {len(val_dataset)}")

    training_losses = []
    validation_losses = []

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    # Define training arguments here
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=logging_steps,
        save_steps=save_steps,
        seed=seed,
        fp16=fp16,
        evaluation_strategy="epoch",
    )

    # Loss tracking callback
    class LossTrackingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                training_losses.append(logs["loss"])
            if logs is not None and "eval_loss" in logs:
                validation_losses.append(logs["eval_loss"])

    if optimizer_choice == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LossTrackingCallback()],
        optimizers=(optimizer, None)
    )

    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    eval_results = trainer.evaluate()

    # Plot training and validation loss graph
    fig, ax = plt.subplots()
    ax.plot(training_losses, label='Training Loss', color='blue')
    if validation_losses:
        ax.plot(validation_losses, label='Validation Loss', color='orange')

    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses')
    ax.legend()
    st.pyplot(fig)

    return eval_results.get("eval_accuracy", None), None, None  # Updated to match your return values

# Function to zip the model directory correctly
def zip_model_directory(model_dir, zip_filename):
    shutil.make_archive(zip_filename, 'zip', model_dir)

# Streamlit UI
st.title("DistilBERT Text Classifier")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    columns_to_drop = st.multiselect("Select columns to drop", data.columns.tolist())

    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        st.write("Updated Data Preview after dropping columns:")
        st.dataframe(data.head())

    st.subheader("Visualize Data")
    x_column = st.selectbox("Select feature column for X-axis", data.columns)
    y_column = st.selectbox("Select target column for Y-axis", [col for col in data.columns if col != x_column])

    if x_column and y_column:
        st.write(f"Scatter plot of {x_column} vs {y_column}")
        fig, ax = plt.subplots()
        if pd.api.types.is_numeric_dtype(data[x_column]) and pd.api.types.is_numeric_dtype(data[y_column]):
            ax.scatter(data[x_column], data[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f"{x_column} vs {y_column}")
        elif pd.api.types.is_numeric_dtype(data[y_column]):
            data.groupby(x_column)[y_column].mean().plot(kind='bar', ax=ax)
            ax.set_ylabel(f"Average of {y_column}")
        else:
            st.write("Selected columns are not suitable for plotting.")
        st.pyplot(fig)

    target_column = st.selectbox("Select target column for training", data.columns)
    num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=3)
    num_rows = st.number_input("Number of Rows for Training", min_value=1, max_value=data.shape[0], value=100)
    optimizer_choice = st.selectbox("Select optimizer", ['AdamW', 'SGD', 'Adam', 'Adagrad'])
    smote_technique = st.selectbox("Select SMOTE technique", ['None', 'SMOTE', 'BorderlineSMOTE', 'SMOTETomek', 'SMOTECut', 'SMOTEENN'])

    # Additional fine-tuning parameters
    learning_rate = st.number_input("Learning Rate", min_value=1e-7, max_value=1e-2, value=5e-5, format="%e")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
    warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=1000, value=0)
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.5, value=0.01)
    adam_epsilon = st.number_input("Adam Epsilon", min_value=1e-8, max_value=1e-5, value=1e-8, format="%e")
    max_grad_norm = st.number_input("Max Gradient Norm", min_value=0.1, max_value=10.0, value=1.0)
    save_steps = st.number_input("Save Steps", min_value=1, max_value=1000, value=500)
    logging_steps = st.number_input("Logging Steps", min_value=1, max_value=1000, value=100)
    seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42)
    fp16 = st.checkbox("Use FP16 Training")

    if st.button("Train Model"):
        train_texts = data[x_column].astype(str).tolist()
        train_labels = data[target_column].tolist()

        # Show training status
        with st.spinner('Training in progress...'):
            eval_accuracy, preds, true_labels = train_distilbert_model(
                train_texts, train_labels, num_epochs, optimizer_choice, learning_rate, batch_size,
                warmup_steps, weight_decay, adam_epsilon, max_grad_norm, save_steps, logging_steps, seed, fp16,
                smote_technique=smote_technique
            )

        st.write(f"Training Accuracy: {eval_accuracy:.2f}")

        # After training, zip the model
        zip_filename = './trained_model'  # This is the directory name
        zip_model_directory('./model', zip_filename)

        # Correct the download to .zip extension
        with open(zip_filename + ".zip", "rb") as f:
            st.download_button("Download Trained Model", f, file_name="trained_model.zip", mime="application/zip")