import hashlib
import numpy as np
from datetime import datetime
import logging

class SimpleBlockchain:
    def __init__(self):
        self.chain = []
        self.logger = logging.getLogger("Blockchain")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler("logs/blockchain.log"))
        self.logger.info("Initialized blockchain")

    def add_block(self, round_id, model_weights):
        try:
            if not isinstance(model_weights, list):
                raise ValueError("Model weights must be a list of numpy arrays")
            
            block_hash = self._hash_weights(model_weights)
            block = {
                'timestamp': datetime.now().isoformat(),
                'round': round_id,
                'hash': block_hash,
                'previous_hash': self.chain[-1]['hash'] if self.chain else '0'
            }
            self.chain.append(block)
            self.logger.info(f"Added global model for round {round_id}")
            return block_hash
        except Exception as e:
            self.logger.error(f"Block addition failed: {str(e)}")
            raise

    def get_latest_hash(self):
        return self.chain[-1]['hash'] if self.chain else None

    def _hash_weights(self, weights):
        hash_str = ""
        for layer in weights:
            layer = np.array(layer)
            hash_str += hashlib.sha256(layer.tobytes()).hexdigest()
        return hashlib.sha256(hash_str.encode()).hexdigest()