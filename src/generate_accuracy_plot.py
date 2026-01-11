import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Recreate same splitting and compute accuracies for ±5%, ±2%, ±1%

def load_data():
    df = pd.read_csv('data/soil_spectral_data_individual.csv')
    unique_combinations = df.groupby(['water_content_percent', 'replicate']).size().reset_index()[['water_content_percent', 'replicate']]
    X_list = []
    y_list = []
    for idx, row in unique_combinations.iterrows():
        wc = row['water_content_percent']
        rep = row['replicate']
        measurement_data = df[(df['water_content_percent'] == wc) & (df['replicate'] == rep)]
        measurement_data = measurement_data.sort_values('wavelength_nm')
        spectral_values = measurement_data['reflectance'].values
        X_list.append(spectral_values)
        y_list.append(wc)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def compute_accuracies(y_true, y_pred):
    abs_err = np.abs(y_true - y_pred)
    acc5 = np.sum(abs_err <= 5.0) / len(y_true) * 100
    acc2 = np.sum(abs_err <= 2.0) / len(y_true) * 100
    acc1 = np.sum(abs_err <= 1.0) / len(y_true) * 100
    return acc5, acc2, acc1


if __name__ == '__main__':
    print('Loading data...')
    X, y = load_data()
    print(f'Loaded {X.shape[0]} samples, {X.shape[1]} features')

    # Reproduce splits used in training script
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True, stratify=y_temp)

    # Load model and scaler
    print('Loading trained model...')
    with open('models/svm_water_content_model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

    # Scale data
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Predict
    y_train_pred = model.predict(X_train_s)
    y_val_pred = model.predict(X_val_s)
    y_test_pred = model.predict(X_test_s)

    # Compute accuracies
    acc_train = compute_accuracies(y_train, y_train_pred)
    acc_val = compute_accuracies(y_val, y_val_pred)
    acc_test = compute_accuracies(y_test, y_test_pred)

    print('\nAccuracies (Train, Val, Test) for ±5%, ±2%, ±1%:')
    print('Train:', acc_train)
    print('Val:  ', acc_val)
    print('Test: ', acc_test)

    # Create bar plot
    labels = ['±5%', '±2%', '±1%']
    train_vals = acc_train
    val_vals = acc_val
    test_vals = acc_test

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width, train_vals, width, label='Train', color='tab:blue')
    ax.bar(x, val_vals, width, label='Validation', color='tab:green')
    ax.bar(x + width, test_vals, width, label='Test', color='tab:orange')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy comparison across Train / Validation / Test (tolerance levels)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()

    for i, v in enumerate(train_vals):
        ax.text(i - width, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(val_vals):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(test_vals):
        ax.text(i + width, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_file = 'results/accuracy_comparison.png'
    plt.savefig(out_file, dpi=300)
    print(f'Plot saved as {out_file}')
