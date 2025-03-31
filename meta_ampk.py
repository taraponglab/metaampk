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

# Define Meta-Model (Dense Neural Network)
def create_meta_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
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
all_metrics_train.append(evaluate_model(meta_model, stacked_train, y_train, 'Meta_CNN_train'))
all_metrics_test.append(evaluate_model(meta_model, stacked_test, y_test, 'Meta_CNN_test'))

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
all_metrics_train_df.to_csv(os.path.join(base_dir, 'CNN_stacked_model_train_metrics.csv'))
all_metrics_test_df.to_csv(os.path.join(base_dir, 'CNN_stacked_model_test_metrics.csv'))
print("All models and metrics have been saved.")

def nearest_neighbor_AD(x_train, x_test, name, k, z=3):
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(x_train)
    joblib.dump(nn, os.path.join(name, f"ad_{k}_{z}.joblib"))
    
    distance, _ = nn.kneighbors(x_train)
    di = np.mean(distance, axis=1)
    dk, sk = np.mean(di), np.std(di)
    distance, _ = nn.kneighbors(x_test)
    di_test = np.mean(distance, axis=1)
    AD_status = ['within_AD' if d < dk + (z * sk) else 'outside_AD' for d in di_test]
    
    df = pd.DataFrame({'AD_status': AD_status}, index=x_test.index)
    return df, dk, sk

def run_ad(meta_model, stacked_test, y_test, name, z=0.5):
    k_values = list(range(3, 11))
    metrics = {
        "k": [], "Accuracy": [], "Balanced Accuracy": [], "Sensitivity": [],
        "Specificity": [], "MCC": [], "AUC": [], "Precision": [], "F1 Score": [],
        "Removed Compounds": [], "dk_values": [], "sk_values": []
    }

    if not isinstance(stacked_test, pd.DataFrame):
        stacked_test = pd.DataFrame(stacked_test)

    for k in k_values:
        t, dk, sk = nearest_neighbor_AD(stacked_test, stacked_test, name, k, z)
        x_ad_test = stacked_test[t['AD_status'] == 'within_AD']
        y_ad_test = y_test[t['AD_status'] == 'within_AD']

        if len(x_ad_test) == 0:
            continue

        y_prob_test = meta_model.predict(x_ad_test)
        y_pred_test = (y_prob_test > 0.5).astype(int)

        acc = accuracy_score(y_ad_test, y_pred_test)
        conf_matrix = confusion_matrix(y_ad_test, y_pred_test)
        tn, fp, fn, tp = conf_matrix.ravel()
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        auc, mcc = roc_auc_score(y_ad_test, y_prob_test), matthews_corrcoef(y_ad_test, y_pred_test)
        bal_acc, precision, f1 = balanced_accuracy_score(y_ad_test, y_pred_test), precision_score(y_ad_test, y_pred_test), f1_score(y_ad_test, y_pred_test)

        metrics["k"].append(k)
        metrics["Accuracy"].append(acc)
        metrics["Balanced Accuracy"].append(bal_acc)
        metrics["Sensitivity"].append(sens)
        metrics["Specificity"].append(spec)
        metrics["MCC"].append(mcc)
        metrics["AUC"].append(auc)
        metrics["Precision"].append(precision)
        metrics["F1 Score"].append(f1)
        metrics["Removed Compounds"].append((t['AD_status'] == 'outside_AD').sum())
        metrics["dk_values"].append(dk)
        metrics["sk_values"].append(sk)

    df_metrics = pd.DataFrame(metrics).round(3)
    df_metrics.to_csv(f"AD_metrics_{name}_{z}.csv", index=False)
    print("AD Metrics saved.")

    # Define values for plotting
    BA_values = metrics["Balanced Accuracy"]
    Sen_values = metrics["Sensitivity"]
    Spe_values = metrics["Specificity"]
    MCC_values = metrics["MCC"]
    AUC_values = metrics["AUC"]
    Precision_values = metrics["Precision"]
    F1_values = metrics["F1 Score"]
    removed_compounds_values = metrics["Removed Compounds"]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Line plots for evaluation metrics
    ax1.plot(k_values, BA_values, 'bo-', label="BACC")
    ax1.plot(k_values, Sen_values, 'gs-', label="Sensitivity")
    ax1.plot(k_values, Spe_values, 'y*-', label="Specificity")
    ax1.plot(k_values, MCC_values, 'r^-', label="MCC")
    ax1.plot(k_values, AUC_values, 'md-', label="AUC")
    ax1.plot(k_values, Precision_values, 'cD-', label="Precision")
    ax1.plot(k_values, F1_values, 'cX-', label="F1 Score")

    # Formatting ax1
    ax1.set_xlabel('k', fontsize=12, fontstyle='italic', weight="bold")
    ax1.set_ylabel('Scores', fontsize=12, fontstyle='italic', weight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1.02))

    # Bar plot for removed compounds
    ax2.bar(k_values, removed_compounds_values, color='green', edgecolor='black', alpha=0.5, width=0.3)
    ax2.set_xlabel('k', fontsize=12, fontstyle='italic', weight="bold")
    ax2.set_ylabel('Removed Compounds', fontsize=12, fontstyle='italic', weight='bold')
    ax2.set_xticks(k_values)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"AD_{name}_{z}_ampk_AD.svg", bbox_inches='tight')
    plt.close()

    # Run AD and y-randomization
#run_ad(meta_model, stacked_test, y_test, name, z=0.5)
print ("finished AD computation")
# Update y_randomization function calls and contents
def y_randomization(stacked_train, stacked_test, y_train, y_test, all_metrics_train_df, all_metrics_test_df):
    MCC_test = []
    MCC_train = []

    for i in range(1, 101):
        # Check if y_train and y_test are numpy arrays or Pandas Series
        if isinstance(y_train, np.ndarray):
            y_train_shuffled = np.random.permutation(y_train)
        else:
            y_train_shuffled = y_train.sample(frac=1, replace=False, random_state=0)

        if isinstance(y_test, np.ndarray):
            y_test_shuffled = np.random.permutation(y_test)
        else:
            y_test_shuffled = y_test.sample(frac=1, replace=False, random_state=0)

        # Random forest model training
        model = RandomForestClassifier(max_depth=5, max_features=10, n_estimators=300).fit(stacked_train, y_train_shuffled)
        y_pred_MCCext = model.predict(stacked_test)
        y_pred_MCCtrain = model.predict(stacked_train)

        # Calculate Matthews correlation coefficient
        MCCext = matthews_corrcoef(y_test_shuffled, y_pred_MCCext)
        MCC_test.append(MCCext)
        MCCtrain = matthews_corrcoef(y_train_shuffled, y_pred_MCCtrain)
        MCC_train.append(MCCtrain)

    # Plotting
    size = [50]
    sizes = [20]
    x = [all_metrics_train_df.loc['Meta_CNN_train', 'MCC']]
    y = [all_metrics_test_df.loc['Meta_CNN_test', 'MCC']]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0.5, c='black', ls=':')
    ax.axhline(0.5, c='black', ls=':')
    ax.scatter(x, y, s=size, c=['red'], marker='x', label='Our model')
    ax.scatter(MCC_train, MCC_test, c='blue', edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    ax.set_xlabel('$MCC_{Train}$', fontsize=14, fontstyle='italic', weight='bold')
    ax.set_ylabel('$MCC_{Test}$', fontsize=14, fontstyle='italic', weight='bold')
    ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig("Y-randomization_ampk.svg", bbox_inches='tight')
    plt.close()

    print("Y-randomization completed.")

# Call y_randomization with concatenated DataFrames
#y_randomization(stacked_train, stacked_test, y_train, y_test, all_metrics_train_df, all_metrics_test_df)
print("Finished y-randomization.")

def plot_roc_and_confusion_matrices(y_train, y_prob_stacked_train, y_test, y_prob_stacked_test):
    # ROC curve for train dataset
    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_stacked_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    # ROC curve for test dataset
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_stacked_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Highlight title label
    plt.title('Receiver Operating Characteristic', fontsize=24, fontweight='bold', color='black')
    plt.legend(loc="lower right")
    plt.savefig("ROC_Curve_Train_Test.svg", bbox_inches='tight')
    plt.close()
    print("Finished plotting ROC curve for train and test datasets.")

    # Define highlight color for labels
    highlight_color = 'black'

    # Confusion matrix for train dataset
    y_train_pred = (y_prob_stacked_train >= 0.5).astype(int)
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['Class 0', 'Class 1'])

    # Confusion matrix for test dataset
    y_test_pred = (y_prob_stacked_test >= 0.5).astype(int)
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Class 0', 'Class 1'])

    # Plot confusion matrices with highlighted title
    disp_train.plot(cmap='Oranges')
    plt.title('Train Confusion Matrix', fontsize=16, fontweight='bold', color=highlight_color)
    plt.savefig("Confusion_Matrix_Train_ampk.svg", bbox_inches='tight')
    plt.close()

    disp_test.plot(cmap='Oranges')
    plt.title('Test Confusion Matrix', fontsize=16, fontweight='bold', color=highlight_color)
    plt.savefig("Confusion_Matrix_Test_ampk.svg", bbox_inches='tight')
    plt.close()
    print("Finished plotting confusion matrices for train and test datasets.")
# Call the function with actual data
plot_roc_and_confusion_matrices(y_train, y_prob_stacked_train, y_test, y_prob_stacked_test)



