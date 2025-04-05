import json
import matplotlib.pyplot as plt
from blockchain import blockchain

def generate_analysis_report(metrics_history):
    report = {
        "fl_metrics": {
            "final_accuracy": metrics_history['accuracy'][-1],
            "final_loss": metrics_history['loss'][-1],
            "training_time": sum(metrics_history['round_times'])
        },
        "blockchain_metrics": {
            "total_blocks": len(blockchain.chain),
            "avg_block_time": np.mean([b['timestamp'] for b in blockchain.chain]),
            "unique_clients": len(set(b['client_id'] for b in blockchain.chain))
        },
        "system_metrics": {
            "avg_tls": np.mean(metrics_history['tls']),
            "throughput": len(metrics_history['accuracy'])/sum(metrics_history['round_times'])
        }
    }
    
    with open('analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate final comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_history['accuracy'], label='FL Accuracy')
    plt.plot(metrics_history['tls'], label='Latency (TLS)')
    plt.plot([b['client_id'] for b in blockchain.chain], label='Blockchain Growth')
    plt.title('Integrated FL-Blockchain Performance')
    plt.legend()
    plt.savefig('plots/final_analysis.png')