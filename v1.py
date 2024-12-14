import streamlit as st
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.optim import AdamW, SGD, Adam, RMSprop
import shutil
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Function to handle imbalanced datasets
def handle_imbalance(train_texts, train_labels, technique, tokenizer=None):
    # Tokenize text into numeric format if tokenizer is provided
    if tokenizer:
        encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='np')
        train_texts = encodings["input_ids"]  # Use input_ids for SMOTE

    # Calculate the minimum number of samples per class
    unique_classes, class_counts = np.unique(train_labels, return_counts=True)
    min_samples = class_counts.min()

    # Set n_neighbors to be min_samples - 1 (but not less than 1)
    n_neighbors = max(1, min_samples - 1)

    if technique == 'SMOTE':
        smote = SMOTE(k_neighbors=n_neighbors)
        train_texts, train_labels = smote.fit_resample(train_texts, train_labels)
    elif technique == 'ADASYN':
        adasyn = ADASYN(n_neighbors=n_neighbors)
        train_texts, train_labels = adasyn.fit_resample(train_texts, train_labels)
    elif technique == 'Random Oversampling':
        ros = RandomOverSampler()
        train_texts, train_labels = ros.fit_resample(train_texts, train_labels)
    elif technique == 'Random Undersampling':
        rus = RandomUnderSampler()
        train_texts, train_labels = rus.fit_resample(train_texts, train_labels)

    return train_texts, train_labels


def train_electra_model(train_texts, train_labels, num_epochs, optimizer_choice,
                        learning_rate, batch_size, warmup_steps, weight_decay,
                        adam_epsilon, max_grad_norm, save_steps, logging_steps,
                        seed, fp16, evaluation_strategy, metrics):
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)

    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator",
                                                             num_labels=len(label_encoder.classes_))

    encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_encodings, val_encodings, train_labels_split, val_labels_split = train_test_split(
        encodings["input_ids"], train_labels, test_size=0.2, random_state=seed
    )

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

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        metrics_results = {'accuracy': accuracy_score(labels, preds)}
        if 'f1' in metrics:
            metrics_results['f1'] = f1_score(labels, preds, average='weighted')
        if 'precision' in metrics:
            metrics_results['precision'] = precision_score(labels, preds, average='weighted')
        if 'recall' in metrics:
            metrics_results['recall'] = recall_score(labels, preds, average='weighted')
        return metrics_results

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        logging_dir='./logs',
        logging_steps=logging_steps,
        save_steps=save_steps,
        seed=seed,
        fp16=fp16,
        evaluation_strategy=evaluation_strategy,
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    class ProgressCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            progress = state.epoch / num_epochs
            progress_bar.progress(progress)
            status_text.text(f"Training Progress: {progress * 100:.2f}%")

    if optimizer_choice == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback()],
        optimizers=(optimizer, None)
    )

    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    eval_results = trainer.evaluate()

    trainer.model.eval()
    with torch.no_grad():
        val_inputs = val_encodings.to(trainer.model.device)
        logits = trainer.model(val_inputs)["logits"]
        preds = np.argmax(logits.cpu().numpy(), axis=1)

    return eval_results.get("eval_accuracy", None), preds, val_labels_split

# Streamlit UI
st.title("META")

# Quick Train button at the top
if st.button("Quick Train"):
    num_epochs = 3
    optimizer_choice = 'AdamW'
    learning_rate = 5e-5
    batch_size = 8
    warmup_steps = 0
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    save_steps = 500
    logging_steps = 10
    seed = 42
    fp16 = False
    evaluation_strategy = 'epoch'  # Default value
    metrics = ['accuracy']  # Default metric
    st.write("Using default hyperparameters for quick training.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Add option to fill NaN values
    nan_fill_option = st.selectbox("Choose how to fill NaN values", ["Mode", "Custom Value"])
    if nan_fill_option == "Mode":
        data = data.fillna(data.mode().iloc[0])  # Fill NaN with mode of each column
        st.write("NaN values filled with mode.")
    elif nan_fill_option == "Custom Value":
        custom_value = st.text_input("Enter custom value to fill NaN", "")
        if custom_value:
            data = data.fillna(custom_value)
            st.write(f"NaN values filled with custom value: {custom_value}")

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

    # Calculate class distribution
    class_distribution = data[target_column].value_counts()
    st.write("Class Distribution:")
    st.bar_chart(class_distribution)
    imbalance_technique = "None"
    # Check for imbalance
    imbalance_threshold =  0.1  # Define a threshold for imbalance
    if class_distribution.min() / class_distribution.sum() < imbalance_threshold:
        st.warning(
            "Warning: There is a significant class imbalance in your dataset. Consider using techniques like SMOTE, ADASYN, or other resampling methods to handle this.")
        imbalance_technique = st.selectbox("Select Technique for Handling Imbalance",
                                        ["None", "SMOTE", "ADASYN", "Random Oversampling", "Random Undersampling"])
    num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=3)
    num_rows = st.number_input("Number of Rows for Training", min_value=1, max_value=data.shape[0], value=100)
    optimizer_choice = st.selectbox("Select optimizer", ['AdamW', 'SGD', 'Adam', 'RMSprop'])

    # Additional fine-tuning parameters
    learning_rate = st.number_input("Learning Rate", min_value=1e-7, max_value=1e-2, value=5e-5, format="%e")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
    warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=1000, value=0)
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.5, value=0.01)
    adam_epsilon = st.number_input("Adam Epsilon", min_value=1e-8, max_value=1e-5, value=1e-8, format="%e")
    max_grad_norm = st.number_input("Max Gradient Norm", min_value=0.1, max_value=10.0, value=1.0)
    save_steps = st.number_input("Save Steps", min_value=1, max_value=1000, value=500)
    logging_steps = st.number_input("Logging Steps", min_value=1, max_value=1000, value=10)
    seed = st.number_input("Random Seed", min_value=0, max_value=1000, value=42)
    fp16 = st.checkbox("Enable FP16 (Mixed Precision)")

    # New options for user selection
    evaluation_strategy = st.selectbox("Select Evaluation Strategy", ['no', 'steps', 'epoch'], index=1)
    metrics = st.multiselect("Select Evaluation Metrics", ['accuracy', 'f1', 'precision', 'recall'], default=['accuracy'])

    st.write(f"Dataset size: {data.shape[0]} rows")

    if st.button("Train Model"):
        if data.shape[0] > num_rows:
            data = data.sample(n=num_rows, random_state=42)

        # Assuming 'data' is your DataFrame and 'target_column' is the name of your target column
        train_texts = data.drop(columns=[target_column]).astype(str).agg(' '.join, axis=1).tolist()
        train_labels = data[target_column].tolist()

        # Ensure train_texts is a list of strings
        if isinstance(train_texts, np.ndarray):
            train_texts = train_texts.tolist()  # Convert to list if it's a NumPy array

        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)

        if imbalance_technique != "None":
            st.write(f"Using {imbalance_technique} to handle class imbalance.")
            train_texts, train_labels = handle_imbalance(
                train_texts, train_labels, imbalance_technique,
                tokenizer=ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
            )

        with st.spinner("Training the model..."):
            final_accuracy, preds, actuals = train_electra_model(
                train_texts, train_labels, num_epochs, optimizer_choice,
                learning_rate=learning_rate,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                adam_epsilon=adam_epsilon,
                max_grad_norm=max_grad_norm,
                save_steps=save_steps,
                logging_steps=logging_steps,
                seed=seed,
                fp16=fp16,
                evaluation_strategy=evaluation_strategy,
                metrics=metrics
            )
            st.success("Model trained successfully!")
            st.write(
                f"Final Accuracy: {final_accuracy:.2f}" if final_accuracy is not None else "Accuracy not available.")

            st.write("Predicted vs Actual Values:")
            prediction_df = pd.DataFrame({'Actual': actuals, 'Predicted': preds})
            prediction_df['Index'] = prediction_df.index
            fig, ax = plt.subplots()
            ax.plot(prediction_df['Index'], prediction_df['Actual'], label='Actual', marker='o')
            ax.plot(prediction_df['Index'], prediction_df['Predicted'], label='Predicted', marker='x')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Class Label')
            ax.set_title('Predicted vs Actual Values')
            ax.legend()
            st.pyplot(fig)

    if st.button("Download Model"):
        shutil.make_archive('model', 'zip', './model')
        with open('model.zip', 'rb') as f:
            st.download_button('Download Trained Model', f, file_name='model.zip', mime='application/zip')