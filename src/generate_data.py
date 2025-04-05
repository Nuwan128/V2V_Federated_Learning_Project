import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_generation.log'),
        logging.StreamHandler()
    ]
)

def generate_client_data(client_id, num_train=800, num_test=200):
    """Generate synthetic vehicle data for V2V communication"""
    np.random.seed(42 + client_id)
    bases = {0: (100, 100), 1: (500, 500), 2: (300, 300)}
    
    def _create_features(n):
        return {
            'speed': np.random.uniform(0, 120, n),
            'pos_x': bases[client_id][0] + np.random.normal(0, 5, n),
            'pos_y': bases[client_id][1] + np.random.normal(0, 5, n),
            'heading': np.random.uniform(0, 360, n),
            'acceleration': np.random.uniform(-5, 5, n),
            'distance_to_nearest': np.random.exponential(scale=50, size=n)
        }

    try:
        total_samples = num_train + num_test
        df = pd.DataFrame(_create_features(total_samples))
        
        # Create collision risk target with enhanced balance
        df['target'] = (
            (df['speed'] > 80) & 
            (df['acceleration'].abs() > 3) &
            (df['distance_to_nearest'] < 40)
        ).astype(int)
        
        # Force minimum 30% positive samples
        positive_mask = df['target'] == 1
        if positive_mask.mean() < 0.3:
            needed = int(total_samples * 0.3) - positive_mask.sum()
            if needed > 0:
                candidates = df[~positive_mask].sample(min(needed, len(df[~positive_mask])))
                df.loc[candidates.index, 'target'] = 1
                positive_mask = df['target'] == 1  # Update mask

        # Verify we have enough positives for stratification
        min_positives = max(1, int(num_train * 0.05))  # At least 5% of train size
        if positive_mask.sum() < min_positives:
            raise ValueError(f"Insufficient positive samples: {positive_mask.sum()}")
        
        # Stratified split using sklearn
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(
            df,
            test_size=num_test,
            stratify=df['target'],
            random_state=42
        )
        
        # Save datasets
        data_dir = Path(f"data/client_{client_id}")
        data_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(data_dir/'train.csv', index=False)
        test.to_csv(data_dir/'test.csv', index=False)
        
        logging.info(
            f"Client {client_id}: {len(train)} train, {len(test)} test. "
            f"Positives - Train: {train['target'].mean():.2%}, Test: {test['target'].mean():.2%}"
        )
        
    except Exception as e:
        logging.error(f"Failed client {client_id}: {str(e)}")
        raise

if __name__ == "__main__":
    for cid in range(3):
        generate_client_data(cid)