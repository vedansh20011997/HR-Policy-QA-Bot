import sys
sys.path.append('../')
from src.indexer.es_indexer import ElasticSearchIndexer
from typing import List

class ElasticSearchRetreiver(ElasticSearchIndexer):
    def __init__(
        self,
        es_cloud_id: str,
        es_api_key: str,
        openai_api_key: str,
        index_name: str = "hr_policies",
        embedding_model: str = "text-embedding-ada-002",
        splade_model_name: str = "naver/splade-cocondenser-ensembledistil",
        source: List[str] = ['content', 'policy_name', 'metadata'],
        rrf_constant: int = 60,
        results_size: int = 10
    ):
        """
        Initialize the Elasticsearch indexer with hybrid search capabilities.
        
        Args:
            es_cloud_id (str): Elasticsearch Cloud ID
            es_api_key (str): Elasticsearch API Key
            openai_api_key (str): OpenAI API Key
            index_name (str): Name of the Elasticsearch index
            embedding_model (str): OpenAI embedding model to use
            splade_model_name (str): SPLADE model name from HuggingFace
            rrf_constant (int): RRF normalization constant
            results_size (int): total tuples to fetch from es
        """
        self.source = source
        self.RRF_CONSTANT = rrf_constant
        self.size = results_size
        super().__init__(es_cloud_id, es_api_key, openai_api_key, index_name, embedding_model, splade_model_name)
        
    def create_policy_filter_query(self, policy_name):
        base_query = {
            "bool": {
                "must": []
            }
        }
        if policy_name != "unknown":
            if policy_name:
                base_query["bool"]["must"].append({
                    "term": {"policy_name": policy_name}
                })
        return base_query
    
    def create_BM25_query(self, base_query, query):
        if len(base_query["bool"]["must"]):
            search_query = {
                    "bool": {
                        "must": [
                            {"match": {"content": query}}
                        ]
                    }
                }
        else:
            search_query = {
                    "bool": {
                        "must": [
                            base_query["bool"]["must"][0],
                            {"match": {"content": query}}
                        ]
                    }
                }
        return search_query
    
    def create_dense_retreival_query(self, base_query, query):
        embedding = self.generate_embeddings(query)
        search_query = {
            "script_score": {
                "query": base_query,
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding}
                    }}
            }
        return search_query
    
    def create_sparse_retreival_query(self, base_query, query):
        splade_vector = self.generate_splade_vector(query)

        # Convert SPLADE vector to rank_feature queries
        rank_features = []
        for term, weight in splade_vector.items():
            rank_features.append({
                "rank_feature": {
                    "field": f"splade_vector.{term}",
                    "boost": weight,
                    "saturation": {
                        "pivot": 1.0
                    }
                }
            })
        
        search_query = {
            "bool": {
                "must": base_query["bool"]["must"],
                "should": rank_features
            }
        }
        return search_query
    
    def implement_RRF_for_hybrid(self, rankwise_results_1, rankwise_results_2):
        final_scores = {}
        results_tuple = {"hits": {"hits": []}}
        all_docs = set()

        combined_results = rankwise_results_1["hits"]["hits"] + rankwise_results_2["hits"]["hits"]
        
        for hit in combined_results:
            all_docs.add(hit["_id"])
        doc_id_to_tuple = {x['_id']: x for x in combined_results}
        
        for doc_id in all_docs:
            score = 0
            # Add RRF score from results 1
            for rank, hit in enumerate(rankwise_results_1["hits"]["hits"]):
                if hit["_id"] == doc_id:
                    score += 1 / (self.RRF_CONSTANT + rank + 1)
                    break
            # Add RRF score from results 2
            for rank, hit in enumerate(rankwise_results_2["hits"]["hits"]):
                if hit["_id"] == doc_id:
                    score += 1 / (self.RRF_CONSTANT + rank + 1)
                    break
            final_scores[doc_id] = score
        
        # Sort by final RRF score
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        for (doc_id, score) in sorted_results:
            tuple_ = doc_id_to_tuple[doc_id]
            tuple_['_score'] = score
            results_tuple["hits"]["hits"].append(tuple_)
        return results_tuple

    def search(self, query: str, policy_name: str = None, method: str = "hybrid_dense_splade"):
        """
        Search using different methods
        method options: bm25, dense, splade, hybrid_dense_splade, hybrid_dense_bm25
        """
        # Base query with policy filter if provided
        base_query = self.create_policy_filter_query(policy_name=policy_name)
        
        # Different search methods
        if method == "bm25":
            search_query = self.create_BM25_query(base_query=base_query, query=query)
        
        elif method == "dense":
            search_query = self.create_dense_retreival_query(base_query=base_query, query=query)
        
        elif method == "splade":
            search_query = self.create_sparse_retreival_query(base_query=base_query, query=query)
        
        elif method == "hybrid_dense_splade":
            # RRF-based combination of dense and SPLADE
            dense_results = self.search(query, policy_name, "dense")
            splade_results = self.search(query, policy_name, "splade")
            
            # Combine results using RRF
            results_tuple = self.implement_RRF_for_hybrid(dense_results, splade_results)
            return results_tuple
        
        elif method == "hybrid_dense_bm25":
            # Similar RRF-based combination of dense and BM25
            dense_results = self.search(query, policy_name, "dense")
            bm25_results = self.search(query, policy_name, "bm25")

            # Combine results using RRF
            results_tuple = self.implement_RRF_for_hybrid(dense_results, bm25_results)
            return results_tuple
        
        # Execute search
        response = self.es.search(
            index=self.index_name,
            query=search_query,
            source = self.source,
            size=self.size
        )
        
        return response

# Usage example
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    search = ElasticSearchRetreiver(
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        es_api_key=os.getenv("ELASTIC_API_KEY"),
        openai_api_key=os.getenv("openai_api_key"),
        index_name=os.getenv("index_name")
    )

    # Search using different methods
    query = "What is the leave policy?"
    policy_filter = "leave policy"
    
    # BM25 search
    bm25_results = search.search(query, policy_filter, "bm25")
    print("bm25_results\n", [x['_source']['content'] for x in bm25_results['hits']['hits']])
    
    # Dense vector search
    dense_results = search.search(query, policy_filter, "dense")
    print("dense_results\n", [x['_source']['content'] for x in dense_results['hits']['hits']])
    
    # SPLADE search
    splade_results = search.search(query, policy_filter, "splade")
    print("splade_results\n", [x['_source']['content'] for x in splade_results['hits']['hits']])
    
    # Hybrid search (Dense + SPLADE)
    hybrid_results = search.search(query, policy_filter, "hybrid_dense_splade")
    print("hybrid_results\n", [x['_source']['content'] for x in hybrid_results['hits']['hits']])

    # Hybrid search (Dense + BM25)
    hybrid_dense_bm25 = search.search(query, policy_filter, "hybrid_dense_bm25")
    print("hybrid_results\n", [x['_source']['content'] for x in hybrid_dense_bm25['hits']['hits']])