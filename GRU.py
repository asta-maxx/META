import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

# Streamlit Title
st.title("GRU Model")

# Sidebar - Dataset Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(df)

    # Handle Missing or Invalid Values
    df.replace("?", np.nan, inplace=True)  # Replace '?' with NaN
    if df.isnull().sum().any():
        st.warning("The dataset contains missing or invalid values. They will be handled automatically.")
        imputer = SimpleImputer(strategy="mean")  # Replace missing values with column mean
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Descriptive Statistics
    st.header("Descriptive Statistics")
    st.write("Central Tendency (Mean, Median):")
    st.write(df.describe().loc[['mean', '50%']])
    st.write("Variability (Standard Deviation):")
    st.write(df.describe().loc[['std']])
    st.write("Data Shape (Skewness, Kurtosis):")
    st.write("Skewness:")
    st.write(df.skew())
    st.write("Kurtosis:")
    st.write(df.kurt())

    # Feature and Label Selection
    st.sidebar.header("Feature and Label Selection")
    feature_columns = st.sidebar.multiselect("Select Feature Columns", df.columns.tolist(), default=df.columns[:-1])
    label_column = st.sidebar.selectbox("Select Label Column", df.columns.tolist(), index=len(df.columns) - 1)
    
    if feature_columns and label_column:
        X = df[feature_columns].values
        y = df[label_column].values

        # Encode Labels for Classification
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Scaling Features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Splitting Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        st.write(f"Training Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")
    else:
        st.warning("Please select both feature columns and a label column.")
        st.stop()

    # Sidebar - Training Parameters
    st.sidebar.header("Training Parameters")
    default_units = st.sidebar.number_input("Default GRU Units", min_value=8, max_value=256, value=64, step=8)
    default_learning_rate = st.sidebar.number_input("Default Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=10)
    batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, value=32)

    # GRU Model Function
    def build_model(units=64, learning_rate=0.001):
        model = Sequential([
            GRU(units=units, input_shape=(X_train.shape[1], 1)),
            Dense(1, activation='sigmoid')  # For binary classification
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Reshape Data for GRU
    X_train_reshaped = np.expand_dims(X_train, axis=2)
    X_test_reshaped = np.expand_dims(X_test, axis=2)

    # Train Button
    if st.sidebar.button("Train Model"):
        st.write("Training Model...")

        # Wrap Model with scikeras KerasClassifier
        model = KerasClassifier(model=build_model, verbose=0, units=default_units, learning_rate=default_learning_rate, epochs=epochs, batch_size=batch_size)

        # Fit the Model
        history = model.fit(X_train_reshaped, y_train)

        # Display Metrics
        train_score = model.score(X_train_reshaped, y_train)
        test_score = model.score(X_test_reshaped, y_test)
        st.write(f"Training Accuracy: {train_score:.4f}")
        st.write(f"Test Accuracy: {test_score:.4f}")

        # Classification Report
        y_pred = model.predict(X_test_reshaped)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Plot Training Progress
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history_['accuracy'], label='Training Accuracy')
        ax[0].set_title("Model Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        ax[1].plot(history.history_['loss'], label='Training Loss')
        ax[1].set_title("Model Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        st.pyplot(fig)

    # Hyperparameter Optimization
    # Hyperparameter Optimization Section
st.sidebar.header("Hyperparameter Optimization")
if st.sidebar.button("Run Hyperparameter Tuning"):
    st.write("Running Hyperparameter Optimization... This may take a while.")

    # Define Hyperparameter Grid
    param_grid = {
        'model__units': [32, 64, 128],
        'model__learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'epochs': [10, 20]
    }

    # Wrap Model for GridSearch
    model = KerasClassifier(model=build_model, verbose=0)

    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Total Number of Iterations
    total_iterations = len(param_grid['model__units']) * len(param_grid['model__learning_rate']) * len(param_grid['batch_size']) * len(param_grid['epochs'])
    current_iteration = 0

    # Grid Search with Custom Loop for Progress
    best_params = None
    best_score = -np.inf

    for units in param_grid['model__units']:
        for lr in param_grid['model__learning_rate']:
            for batch in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    # Update Progress
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)
                    status_text.write(f"Iteration {current_iteration}/{total_iterations}: Units={units}, LR={lr}, Batch={batch}, Epochs={epochs}")

                    # Train Model for Current Parameters
                    model = KerasClassifier(
                        model=build_model,
                        verbose=0,
                        units=units,
                        learning_rate=lr,
                        batch_size=batch,
                        epochs=epochs
                    )

                    try:
                        model.fit(X_train_reshaped, y_train, verbose=0)
                        score = model.score(X_test_reshaped, y_test)

                        # Track Best Parameters
                        if score > best_score:
                            best_score = score
                            best_params = {'units': units, 'learning_rate': lr, 'batch_size': batch, 'epochs': epochs}

                    except Exception as e:
                        st.warning(f"Failed for {units}, {lr}, {batch}, {epochs}. Error: {e}")

    # Final Results
    progress_bar.progress(1.0)
    status_text.write("Hyperparameter Tuning Completed!")
    st.write("Best Parameters:")
    st.json(best_params)
    st.write(f"Best Accuracy: {best_score:.4f}")


else:
    st.write("Please upload a CSV file to get started.")
