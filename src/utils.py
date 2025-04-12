import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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

def plot_confusion_matrix(y_true, y_pred, client_id=None, save_path="plots"):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Collision', 'Collision'])
    disp.plot(cmap=plt.cm.Blues)
    title = f"Confusion Matrix {'(Client ' + str(client_id) + ')' if client_id is not None else ''}"
    plt.title(title)
    plt.savefig(f"{save_path}/confusion_matrix{'_client_' + str(client_id) if client_id is not None else ''}.png")
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, client_id=None, save_path="plots"):
    """Generate and save precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title = f"Precision-Recall Curve {'(Client ' + str(client_id) + ')' if client_id is not None else ''}"
    plt.title(title)
    plt.legend()
    plt.savefig(f"{save_path}/pr_curve{'_client_' + str(client_id) if client_id is not None else ''}.png")
    plt.close()

def plot_roc_curve(y_true, y_scores, client_id=None, save_path="plots"):
    """Generate and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = f"ROC Curve {'(Client ' + str(client_id) + ')' if client_id is not None else ''}"
    plt.title(title)
    plt.legend()
    plt.savefig(f"{save_path}/roc_curve{'_client_' + str(client_id) if client_id is not None else ''}.png")
    plt.close()

def plot_metric_distribution(metric_values, metric_name, save_path="plots"):
    """Generate histogram of metric distribution across rounds or clients"""
    plt.figure()
    plt.hist(metric_values, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric_name}')
    plt.savefig(f"{save_path}/{metric_name.lower().replace(' ', '_')}_distribution.png")
    plt.close()