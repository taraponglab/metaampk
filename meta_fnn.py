import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score, balanced_accuracy_score, precision_score, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from joblib import dump
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load training and testing data
name = "deep learning model"
fingerprint_types = ["xat", "xes", "xke", "xpc", "xss", "xcd", "xcn", "xkc", "xce", "xsc", "xac", "xma"]

X_train_dict, X_test_dict = {}, {}

for ftype in fingerprint_types:
    X_train_dict[ftype] = np.array(pd.read_csv(os.path.join(name, "train", f'{ftype}_train.csv'), index_col=0).values, dtype=float)
    X_test_dict[ftype] = np.array(pd.read_csv(os.path.join(name, "test", f'{ftype}_test.csv'), index_col=0).values, dtype=float)

y_train = np.array(pd.read_csv(os.path.join(name, "train", "y_train.csv"), index_col=0).values.ravel(), dtype=float)
y_test = np.array(pd.read_csv(os.path.join(name, "test", "y_test.csv"), index_col=0).values.ravel(), dtype=float)

# Model evaluation function
def evaluate_model(model, x_data, y_data, col_name):
    y_prob = model.predict(x_data)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_data, y_pred)
    sen = recall_score(y_data, y_pred)
    mcc = matthews_corrcoef(y_data, y_pred)
    f1  = f1_score(y_data, y_pred)
    auc = roc_auc_score(y_data, y_prob)
    bcc = balanced_accuracy_score(y_data, y_pred)
    pre = precision_score(y_data, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_data, y_pred).ravel()
    spc = tn / (tn + fp)
    
    return pd.DataFrame({
        'Accuracy': [acc], 'Sensitivity': [sen], 'Specificity': [spc],
        'MCC': [mcc], 'F1 Score': [f1], 'AUC': [auc], 'BACC': [bcc], 'Precision': [pre]
    }, index=[col_name])

# Define BiLSTM model
def create_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define CNN model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define Meta-Model (FNN meta)
def create_meta_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Ensure to save models in a base directory
base_dir = "models"

# Store all metrics and predictions
all_metrics_train, all_metrics_test = [], []
all_predictions_bilstm_train, all_predictions_cnn_train = [], []
all_predictions_bilstm_test, all_predictions_cnn_test = [], []

# Train and evaluate models per fingerprint type
for ftype in fingerprint_types:
    X_train_np = X_train_dict[ftype].reshape((X_train_dict[ftype].shape[0], X_train_dict[ftype].shape[1], 1))
    X_test_np = X_test_dict[ftype].reshape((X_test_dict[ftype].shape[0], X_test_dict[ftype].shape[1], 1))
    
    input_shape = (X_train_np.shape[1], 1)

    # Create subdirectory for each fingerprint type
    ftype_dir = os.path.join(base_dir, ftype)
    if not os.path.exists(ftype_dir):
        os.makedirs(ftype_dir)
    
    # Train BiLSTM
    bilstm_model = create_bilstm_model(input_shape)
    bilstm_model.fit(X_train_np, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    # Save the BiLSTM model
    bilstm_model.save(os.path.join(ftype_dir, f'bilstm_model_{ftype}.h5'))
    
    all_metrics_train.append(evaluate_model(bilstm_model, X_train_np, y_train, f'BiLSTM_{ftype}_train'))
    all_metrics_test.append(evaluate_model(bilstm_model, X_test_np, y_test, f'BiLSTM_{ftype}_test'))
    
    all_predictions_bilstm_train.append(bilstm_model.predict(X_train_np))
    all_predictions_bilstm_test.append(bilstm_model.predict(X_test_np))
    
    # Train CNN
    cnn_model = create_cnn_model(input_shape)
    cnn_model.fit(X_train_np, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    # Save the CNN model
    cnn_model.save(os.path.join(ftype_dir, f'cnn_model_{ftype}.h5'))
    
    all_metrics_train.append(evaluate_model(cnn_model, X_train_np, y_train, f'CNN_{ftype}_train'))
    all_metrics_test.append(evaluate_model(cnn_model, X_test_np, y_test, f'CNN_{ftype}_test'))
    
    all_predictions_cnn_train.append(cnn_model.predict(X_train_np))
    all_predictions_cnn_test.append(cnn_model.predict(X_test_np))

# Stack the predictions for the meta-model
stacked_train = np.hstack([np.concatenate(all_predictions_bilstm_train, axis=1),
                           np.concatenate(all_predictions_cnn_train, axis=1)])
stacked_test = np.hstack([np.concatenate(all_predictions_bilstm_test, axis=1),
                          np.concatenate(all_predictions_cnn_test, axis=1)])
pd.DataFrame(stacked_train).to_csv("stacked_train.csv", index=True)
pd.DataFrame(stacked_test).to_csv("stacked_test.csv", index=True)

# Train meta-model
meta_model = create_meta_model(stacked_train.shape[1])
meta_model.fit(stacked_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Save the meta-model
meta_model.save(os.path.join(base_dir, 'meta_model.h5'))

# Evaluate meta-model
all_metrics_train.append(evaluate_model(meta_model, stacked_train, y_train, 'Meta_FNN_train'))
all_metrics_test.append(evaluate_model(meta_model, stacked_test, y_test, 'Meta_FNN_test'))

# Generate predictions using the meta-model
y_prob_stacked_test = meta_model.predict(stacked_test)
y_pred_stacked_test = (y_prob_stacked_test > 0.5).astype(int)
y_prob_stacked_train = meta_model.predict(stacked_train)
y_pred_stacked_train = (y_prob_stacked_train > 0.5).astype(int)
## Ensure arrays are 1-dimensional
y_prob_stacked_test = np.ravel(y_prob_stacked_test)
y_pred_stacked_test = np.ravel(y_pred_stacked_test)
y_prob_stacked_train = np.ravel(y_prob_stacked_train)
y_pred_stacked_train = np.ravel(y_pred_stacked_train)

# Create DataFrames
df_test = pd.DataFrame({
    'y_prob_stacked_test': y_prob_stacked_test,
    'y_pred_stacked_test': y_pred_stacked_test
})

df_train = pd.DataFrame({
    'y_prob_stacked_train': y_prob_stacked_train,
    'y_pred_stacked_train': y_pred_stacked_train
})

# Save to CSV
df_test.to_csv('stacked_test_predictions.csv', index=False)
df_train.to_csv('stacked_train_predictions.csv', index=False)
print("Predictions saved successfully!")

# Save meta-model probability predictions
np.savetxt(os.path.join(base_dir, "meta_model_y_prob_test.csv"), y_prob_stacked_test, delimiter=",")
np.savetxt(os.path.join(base_dir, "meta_model_y_prob_train.csv"), y_prob_stacked_train, delimiter=",")


# After training and evaluating models, concatenate metrics
all_metrics_train_df = pd.concat(all_metrics_train, axis=0)
all_metrics_test_df = pd.concat(all_metrics_test, axis=0)

# Save all metrics
all_metrics_train_df.to_csv(os.path.join(base_dir, 'rerun_FNN_stacked_model_train_metrics.csv'))
all_metrics_test_df.to_csv(os.path.join(base_dir, 'rerun_FNN_stacked_model_test_metrics.csv'))
print("All models and metrics have been saved.")
