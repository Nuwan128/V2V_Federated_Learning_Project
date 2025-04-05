import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

def load_client_data(client_id):
    logger = logging.getLogger(f"DataLoader-{client_id}")
    try:
        data_dir = Path(f"data/client_{client_id}")
        train = pd.read_csv(data_dir/"train.csv")
        test = pd.read_csv(data_dir/"test.csv")
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train.drop('target', axis=1))
        X_test = scaler.transform(test.drop('target', axis=1))
        
        logger.info(f"Loaded data: {len(X_train)} train, {len(X_test)} test")
        return X_train, train['target'].values, X_test, test['target'].values
    
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return None, None, None, None

def plot_training_history(history, client_id):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Client {client_id} Training History')
    plt.legend()
    plt.savefig(f'plots/client_{client_id}_history.png')
    plt.close()