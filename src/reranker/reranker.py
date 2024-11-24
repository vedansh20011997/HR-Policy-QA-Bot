from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ReRanker:
    def __init__(
            self, 
            crossencoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            reranking_window: int = 10
        ):
        self.reranking_window = reranking_window

        # Initialize Cross-Encoder
        self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(crossencoder_model_name)
        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(crossencoder_model_name)
        self.cross_encoder.eval()  # Set to evaluation mode

    def rerank_with_cross_encoder(self, 
                                  query: str, 
                                  results: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank results using the cross-encoder
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, hit["_source"]["content"]) for hit in results]
        features = self.cross_encoder_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # Get cross-encoder scores
        with torch.no_grad():
            scores = self.cross_encoder(**features).logits.squeeze(-1)
            scores = torch.sigmoid(scores).numpy()

        # Combine results with scores
        scored_results = [(result, float(score)) for result, score in zip(results, scores)]
        
        # Sort by cross-encoder score
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
        
        return reranked_results[:self.reranking_window]
    
    def re_ranker(self, 
                  query: str,
                  results: List[Dict[str, Any]]):
        """
        Reranking top-k results given query
        """

        # Rerank using cross-encoder
        reranked_results = self.rerank_with_cross_encoder(
            query=query,
            results=results["hits"]["hits"]
        )
        
        # Format results
        return {
            "hits": {
                "total": {"value": len(reranked_results)},
                "hits": [
                    {
                        **result[0],
                        "_score": result[1]  # Use cross-encoder score
                    }
                    for result in reranked_results
                ]
            }
        }
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    import sys
    sys.path.append('../')
    
    from retreiver.es_retreiver import ElasticSearchRetreiver
    search = ElasticSearchRetreiver(
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        es_api_key=os.getenv("ELASTIC_API_KEY"),
        openai_api_key=os.getenv("openai_api_key"),
        index_name=os.getenv("index_name")
    )

    # Search using different methods
    query = "What is the leave policy?"
    policy_filter = "leave policy"

    dense_results = search.search(query, policy_filter, "dense")

    rr = ReRanker()
    reranked_results = rr.re_ranker(query=query, results=dense_results)
    print(reranked_results)