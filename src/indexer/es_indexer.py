from datetime import datetime
from elasticsearch import Elasticsearch
import openai
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class ElasticSearchIndexer:
    def __init__(
        self,
        es_cloud_id: str,
        es_api_key: str,
        openai_api_key: str,
        index_name: str = "hr_policies",
        embedding_model: str = "text-embedding-ada-002",
        splade_model_name: str = "naver/splade-cocondenser-ensembledistil"
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
        """
        # Initialize Elasticsearch client
        self.es = Elasticsearch(
            cloud_id=es_cloud_id,
            api_key=es_api_key
        )
        
        # Set up OpenAI
        openai.api_key = openai_api_key
        self.embedding_model = embedding_model
        
        # Initialize SPLADE model
        self.tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
        self.splade_model = AutoModelForMaskedLM.from_pretrained(splade_model_name)
        self.splade_model.eval()
        
        self.index_name = index_name
        
        # Create index if it doesn't exist
        self._create_index()
        
    def _create_index(self):
        """Create Elasticsearch index with appropriate mappings for hybrid search."""
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "metadata": {
                                "properties": {
                                    "timestamp": {"type": "date"},
                                    "pdf_source": {"type": "keyword"}
                            }
                        },
                        "policy_name": {
                            "type": "keyword"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1536,  # OpenAI embedding dimensions
                            "similarity": "cosine"
                        },
                        "splade_vector": {
                            "type": "rank_features"  # For SPLADE sparse vectors
                        }
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate OpenAI embeddings"""
        response = openai.embeddings.create(
            input=text,
            model=self.embedding_model,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def generate_splade_vector(self, text: str) -> Dict[str, float]:
        """Generate SPLADE sparse vector representation."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.splade_model(**inputs)
            # Get SPLADE weights
            logits = outputs.logits[0]
            weights = torch.max(torch.log1p(torch.relu(logits)) * inputs['attention_mask'][0].unsqueeze(-1), dim=0)[0]
            
        # Convert to sparse dictionary
        sparse_dict = {}
        nonzero_indices = torch.nonzero(weights).squeeze()
        for idx in nonzero_indices:
            token = self.tokenizer.decode([idx])
            if '.' in token:
                continue
            if weights[idx] > 0:
                sparse_dict[token] = float(weights[idx])
        
        sparse_dict = dict(sorted(sparse_dict.items(), key=lambda item: item[1], reverse=True)[:100])
        return sparse_dict
    
    def index_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Index a single PDF chunk with its embeddings.
        
        Args:
            chunk (dict): Dictionary containing:
                - content (str): The text content of the chunk
                - metadata (dict): Metadata about the chunk
        
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            # Generate vectors
            embedding = self.generate_embeddings(chunk['page_content'])
            splade_vector = self.generate_splade_vector(chunk['page_content'])
            
            # Prepare document
            doc = {
                "content": chunk['page_content'],
                "policy_name": chunk['metadata']['policy_name'],
                "embedding": list(embedding),
                "splade_vector": splade_vector,
                "metadata": {
                        "pdf_source": chunk['metadata']['pdf_name'],
                        "timestamp": datetime.now().isoformat()
                    },
            }
            
            response = self.es.index(
                    index=self.index_name,
                    document=doc
                )

            return response['result'] == 'created'
    
        except Exception as e:
            print(f"Error indexing chunk: {str(e)}")
            return False
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index multiple PDF chunks.
        
        Args:
            chunks (List[Dict]): List of chunks to index
            
        Returns:
            Dict[str, int]: Summary of indexing results
        """
        results = {
            "successful": 0,
            "failed": 0
        }
        
        for chunk in tqdm(chunks):
            if self.index_chunk(chunk):
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        return results
    
if __name__ == "__main__":
    import os, sys
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.append('../parser_and_chunk_creator')
    from pdf_chunk_creator import CreateChunks
    pdf_directory = os.getenv("PDF_DIRECTORY", "/Users/vedanshsharma/Desktop/Simpplr_assignment/DS_Assignment/data/")
    obj = CreateChunks(pdf_directory=pdf_directory)

    chunked_docs = obj.get_chunks()

    # Initialize indexer
    indexer = ElasticSearchIndexer(
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        es_api_key=os.getenv("ELASTIC_API_KEY"),
        openai_api_key=os.getenv("openai_api_key"),
        index_name="hr_policies_new"
    )
    
    # Example chunks with policy names
    chunks = [
        {"page_content": x.page_content, 
         "metadata": x.metadata} for x in chunked_docs
         ]
    
    # Index chunks
    results = indexer.index_chunks(chunks)
    print(f"Indexing results: {results}")