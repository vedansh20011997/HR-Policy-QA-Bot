import pickle
import numpy as np
from typing import List, Dict, Any
from time import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int
    retrieval_time: float

@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    mrr: float
    latency: float
    throughput: float
    memory_usage: float

class SearchEvaluator:
    def __init__(self, search_system):
        self.search_system = search_system
        
    def calculate_precision_at_k(self, relevant_docs: set, retrieved_docs: List[str], k: int) -> float:
        """Calculate precision@k"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_retrieved = len(relevant_docs.intersection(retrieved_k))
        return relevant_retrieved / k

    def calculate_recall_at_k(self, relevant_docs: set, retrieved_docs: List[str], k: int) -> float:
        """Calculate recall@k"""
        if not relevant_docs or not retrieved_docs:
            return 0.0
            
        retrieved_k = set(retrieved_docs[:k])
        relevant_retrieved = len(relevant_docs.intersection(retrieved_k))
        return relevant_retrieved / len(relevant_docs)

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_mrr(self, relevant_docs: set, retrieved_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        return 0.0

    def evaluate_all_metrics(self, 
                           test_queries: List[Dict[str, Any]],
                           k_values: List[int] = [1, 3, 5, 10]) -> List[Dict[str, EvaluationMetrics]]:
        """
        Comprehensive evaluation of all metrics for different retrieval methods
        
        test_queries format:
        [
            {
                'query': 'search query',
                'relevant_docs': ['doc1', 'doc2'],  # List of relevant document IDs
                'relevance_scores': [3, 2, 1, 0],   # Graded relevance scores for evaluation
            },
            ...
        ]
        """
        results = []
        method_metrics = {k: [] for k in k_values}
        latencies = []
        throughput_samples = []
        
        for query_data in tqdm(test_queries):
            try:
                query = query_data['query']
                relevant_docs = set(query_data['relevant_docs'])
                
                # Measure search performance
                start_time = time()
                search_results = self.search_system.get_answer(query)
                latency = time() - start_time

                retrieved_docs = [hit['_id'] for hit in search_results['sources']]
                retrieved_scores = [hit['_score'] for hit in search_results['sources']]
                
                # Calculate metrics for different k values
                for k in k_values:
                    precision = self.calculate_precision_at_k(relevant_docs, retrieved_docs, k)
                    recall = self.calculate_recall_at_k(relevant_docs, retrieved_docs, k)
                    f1 = self.calculate_f1_score(precision, recall)
                    mrr = self.calculate_mrr(relevant_docs, retrieved_docs)
                    
                    method_metrics[k].append(EvaluationMetrics(
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        mrr=mrr,
                        latency=latency,
                        throughput=1/latency,  # Single query throughput
                        memory_usage=0  # Memory usage tracking would need to be implemented separately
                    ))
                
                latencies.append(latency)
                throughput_samples.append(1/latency)
            except Exception as e:
                print(f'Failure due to error: {e}')
                continue
        
        # Aggregate metrics
        results.append({
            'relevance_metrics': {
                k: {
                    'precision': np.mean([m.precision for m in metrics]),
                    'recall': np.mean([m.recall for m in metrics]),
                    'f1': np.mean([m.f1_score for m in metrics]),
                    'mrr': np.mean([m.mrr for m in metrics])
                } for k, metrics in method_metrics.items()
            },
            'performance_metrics': {
                'latency': {
                    'mean': np.mean(latencies),
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                },
                'throughput': np.mean(throughput_samples)
            }
        })
            
        return results

    def plot_evaluation_results(self, results: List[Dict[str, Any]], output_path: str = 'evaluation_results.pdf'):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot Precision@k
        ax = axes[0, 0]
        for metrics in results:
            k_values = list(metrics['relevance_metrics'].keys())
            precision_values = [metrics['relevance_metrics'][k]['precision'] for k in k_values]
            ax.plot(k_values, precision_values, marker='o')
        ax.set_title('Precision@k')
        ax.set_xlabel('k')
        ax.set_ylabel('Precision')
        ax.legend()
        
        # Plot Latency Distribution
        ax = axes[1, 0]
        latency_data = []
        labels = []
        for metrics in results:
            latency_metrics = metrics['performance_metrics']['latency']
            latency_data.append([latency_metrics['mean'], latency_metrics['p50'], 
                               latency_metrics['p95'], latency_metrics['p99']])
        
        ax.boxplot(latency_data)
        ax.set_title('Latency Distribution')
        ax.set_ylabel('Seconds')
        plt.xticks(rotation=45)
        
        # Plot Throughput Comparison
        ax = axes[1, 1]
        throughputs = [metrics['performance_metrics']['throughput'] for metrics in results]
        # ax.bar(methods, throughputs)
        ax.set_title('Throughput Comparison')
        ax.set_ylabel('Queries per Second')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

# Usage Example
if __name__ == "__main__":
    import sys, os
    sys.path.append('../../')

    from src.generator.generation import QABot

    from dotenv import load_dotenv
    load_dotenv()

    import json
    with open(r"ground_truth_data.json", "r") as read_file:
        data = json.load(read_file)

    data = data[:50] #running evaluation on 50 random questions
    reranking_enum = [True, False]
    prefiltering_enum = [True, False]
    retreival_statergy = ["bm25", "dense", "splade", "hybrid_dense_splade", "hybrid_dense_bm25"]

    logging_dict = []

    for prefiltering_flag in prefiltering_enum:
        for reranking_flag in reranking_enum:
            for method in retreival_statergy:
                logging_key = f'prefiltering_{prefiltering_flag}|reranking_{reranking_flag}|method_{method}'
                bot = QABot(
                            es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
                            es_api_key=os.getenv("ELASTIC_API_KEY"),
                            openai_api_key=os.getenv("openai_api_key"),
                            index_name=os.getenv("index_name"),
                            retreival_statergy=method,
                            pre_filtering_required=prefiltering_flag,
                            reranking_enabled=reranking_flag,
                            few_shots_count=10,
                            generation_required=False
                )
                
                # Initialize evaluator
                evaluator = SearchEvaluator(bot)
                
                # Prepare test queries with relevance judgments
                test_queries = [
                    {
                        'query': x['question'],
                        'relevant_docs': [x['_id']],
                        'relevance_scores': [1]  # Graded relevance
                    }
                    for x in data
                ]
                
                # Run evaluation
                results = evaluator.evaluate_all_metrics(
                    test_queries=test_queries,
                    k_values=[1, 3, 5, 10]
                )
                
                # Plot results
                evaluator.plot_evaluation_results(results, output_path=f'performance_logging/evaluation_results|prefiltering_{prefiltering_flag}|reranking_{reranking_flag}|method_{method}.pdf')
                logging_dict.append({
                    "experiment_id": logging_key,
                    "logging_metrics": results[0]
                })
                with open('performance_logging/logging_dict.pkl', 'wb') as f:
                    pickle.dump(logging_dict, f) 
                print(logging_dict[-1])