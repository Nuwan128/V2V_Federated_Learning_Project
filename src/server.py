import os
import numpy as np
import flwr as fl
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from blockchain import SimpleBlockchain
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SERVER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)

class BlockchainStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3
        )
        self.blockchain = SimpleBlockchain()
        self.current_round = 0
        self.metrics_history = {
            'accuracy': [], 'loss': [],
            'precision': [], 'recall': [],
            'f1': [],
            'tls': [], 'round_times': [],
            'blockchain_rounds': []
        }

    def initialize_parameters(self, client_manager):
        initial_weights = [
            np.random.randn(6, 64).astype(np.float32) * 0.1,
            np.zeros(64).astype(np.float32),
            np.random.randn(64, 32).astype(np.float32) * 0.1,
            np.zeros(32).astype(np.float32),
            np.random.randn(32, 1).astype(np.float32) * 0.1,
            np.zeros(1).astype(np.float32)
        ]
        self.blockchain.add_block(self.current_round, initial_weights)
        self.current_round += 1
        return fl.common.ndarrays_to_parameters(initial_weights)

    def aggregate_fit(self, server_round, results, failures):
        verified_results = []
        current_global_hash = self.blockchain.get_latest_hash()
        
        for client_proxy, fit_res in results:
            client_hash = fit_res.metrics.get("global_hash")
            if client_hash == current_global_hash:
                verified_results.append((client_proxy, fit_res))
                logging.info(f"Verified client {fit_res.metrics['client_id']}")
            else:
                logging.warning(f"Rejected client {fit_res.metrics['client_id']} (stale model)")

        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round, verified_results, failures
        )
        
        if aggregated_parameters:
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            self.blockchain.add_block(self.current_round, aggregated_weights)
            self.current_round += 1
            self.metrics_history['blockchain_rounds'].append(self.current_round)
        
        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        
        # Manually aggregate metrics
        total_samples = sum([r.num_examples for _, r in results])
        agg_loss = np.sum([r.loss * r.num_examples for _, r in results]) / total_samples
        
        # Initialize metrics
        agg_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'latency': 0.0
        }
        
        # Weighted average for each metric
        for _, r in results:
            for metric in agg_metrics.keys():
                if metric in r.metrics:
                    agg_metrics[metric] += r.metrics[metric] * r.num_examples
        
        for metric in agg_metrics.keys():
            agg_metrics[metric] /= total_samples
        
        # Store metrics
        self.metrics_history['loss'].append(agg_loss)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:  # Include F1
            self.metrics_history[metric].append(agg_metrics[metric])
        self.metrics_history['tls'].append(agg_metrics['latency'])
        
        self._generate_plots()
        return agg_loss, agg_metrics

    def _generate_plots(self):
        plt.figure(figsize=(15, 10))
        
        # FL Metrics
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history['accuracy'], label='Accuracy')
        plt.plot(self.metrics_history['loss'], label='Loss')
        plt.title('FL Performance Metrics')
        plt.legend()
        
        # Blockchain Growth
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history['blockchain_rounds'], 'g-', label='Global Rounds')
        plt.title('Blockchain Progression')
        plt.legend()
        
        # Latency Correlation
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history['accuracy'], label='Accuracy')
        plt.plot(self.metrics_history['tls'], label='Latency (s)')
        plt.title('Model Accuracy vs Latency')
        plt.legend()
        
        # Throughput
        plt.subplot(2, 2, 4)
        if self.metrics_history['tls']:
            throughput = [i/sum(self.metrics_history['tls'][:i+1]) for i in range(len(self.metrics_history['tls']))]
            plt.plot(throughput, label='Throughput (rounds/s)')
            plt.title('System Throughput')
            plt.legend()
        
        plt.savefig('plots/combined_metrics.png')
        plt.close()

        if len(self.metrics_history['accuracy']) == 10:  # After 10 rounds
            self._generate_results_table()

    def _generate_results_table(self):
        """Generate professional results table as PNG/PDF"""
        Path("results").mkdir(exist_ok=True)
        
        # Prepare data
        rounds = range(1, 11)
        metrics = {
            'Accuracy': self.metrics_history['accuracy'],
            'Loss': self.metrics_history['loss'],
            'Precision': self.metrics_history['precision'],
            'Recall': self.metrics_history['recall'],
            'F1': self.metrics_history['f1'],
            'Latency (s)': self.metrics_history['tls'],
            'Blockchain Round': self.metrics_history['blockchain_rounds'][:10]  # First 10 global updates
        }

        # Create formatted table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        columns = ('Round', 'Accuracy', 'Loss', 'Precision', 'Recall', 'F1', 'Latency', 'BC Round')
        cell_text = []
        for rnd in rounds:
            cell_text.append([
                f"{rnd}",
                f"{metrics['Accuracy'][rnd-1]:.2%}",
                f"{metrics['Loss'][rnd-1]:.4f}",
                f"{metrics['Precision'][rnd-1]:.2%}",
                f"{metrics['Recall'][rnd-1]:.2%}",
                f"{metrics['F1'][rnd-1]:.2%}",
                f"{metrics['Latency (s)'][rnd-1]:.2f}",
                f"{metrics['Blockchain Round'][rnd-1]}"
            ])

        # Create table
        table = ax.table(
            cellText=cell_text,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=['#f3f3f3']*8
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Save outputs
        plt.savefig('results/results_table.png', bbox_inches='tight')
        with PdfPages('results/results_table.pdf') as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
def main():
    server_address = "0.0.0.0:8080" if os.getenv("DOCKERIZED") else "localhost:8080"
    strategy = BlockchainStrategy()
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    main()