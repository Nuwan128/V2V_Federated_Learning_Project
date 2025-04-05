import os
import flwr as fl
import tensorflow as tf
import logging
import time
import hashlib
import numpy as np
import pandas as pd
from utils import load_client_data

tf.get_logger().setLevel('ERROR')

class V2VClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.logger = logging.getLogger(f"Client-{cid}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(f"logs/client_{cid}.log"))
        
        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(cid)
        self._log_data_distribution()
        self.model = self._build_model()

    def _log_data_distribution(self):
        """Log class distribution for validation"""
        train_counts = np.unique(self.y_train, return_counts=True)
        test_counts = np.unique(self.y_test, return_counts=True)
        self.logger.info(f"Class distribution - Train: {dict(zip(train_counts[0], train_counts[1]))}")
        self.logger.info(f"Class distribution - Test: {dict(zip(test_counts[0], test_counts[1]))}")

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5)
            ]
        )
        return model

    def _hash_weights(self, weights):
        hash_str = ""
        for layer in weights:
            layer = np.array(layer)
            hash_str += hashlib.sha256(layer.tobytes()).hexdigest()
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        start_time = time.time()
        self.model.set_weights(parameters)
        global_hash = self._hash_weights(parameters)
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        return (
            self.model.get_weights(),
            len(self.X_train),
            {
                "client_id": self.cid,
                "global_hash": global_hash,
                "train_time": time.time() - start_time
            }
        )

    def evaluate(self, parameters, config):
        start_time = time.time()
        self.model.set_weights(parameters)
        
        # Get predictions for debugging
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Calculate metrics manually
        loss = float(tf.keras.losses.binary_crossentropy(self.y_test, y_pred).numpy().mean())
        accuracy = float(tf.keras.metrics.binary_accuracy(self.y_test, y_pred).numpy().mean())
        precision = float(tf.keras.metrics.Precision(thresholds=0.5)(self.y_test, y_pred).numpy())
        recall = float(tf.keras.metrics.Recall(thresholds=0.5)(self.y_test, y_pred).numpy())

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    
        
        # Log detailed predictions
        self.logger.info(
            f"Evaluation - Loss: {loss:.4f} | Acc: {accuracy:.2%} | "
            f"Prec: {precision:.2%} | Rec: {recall:.2%} | F1: {f1:.2%}"
        )
        
        return (
            loss,  # Python float
            len(self.X_test),  # int
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency": float(time.time() - start_time)  # Ensure Python float
            }
        )

def main():
    import sys
    cid = int(sys.argv[1])
    server_address = "server:8080" if os.getenv("DOCKERIZED") else "localhost:8080"
    fl.client.start_client(server_address=server_address, client=V2VClient(cid).to_client())

if __name__ == "__main__":
    main()