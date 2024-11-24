from dotenv import load_dotenv
load_dotenv()

from src.retreiver.get_policy_name import PolicyClassifier
from src.retreiver.es_retreiver import ElasticSearchRetreiver
from src.reranker.reranker import ReRanker
from src.generator.system_prompt_template import system_message
import openai, os
from typing import List, Dict

class QABot(PolicyClassifier, ElasticSearchRetreiver, ReRanker):
    def __init__(
            self,
            es_cloud_id: str,
            es_api_key: str,
            openai_api_key: str,
            retreival_statergy: str,
            pre_filtering_required: bool = True,
            index_name: str = "hr_policies",
            embedding_model: str = "text-embedding-ada-002",
            splade_model_name: str = "naver/splade-cocondenser-ensembledistil",
            source: List[str] = ['content', 'policy_name', 'metadata'],
            reranking_enabled: bool = True,
            rrf_constant: int = 60,
            results_size: int = 10,
            crossencoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            reranking_window: int = 10,
            generation_model_name: str = 'gpt-4o',
            generation_temperature: float = 0.1,
            generate_max_tokens: int = 1000,
            few_shots_count: int = 5,
            generation_required: bool = True) -> None:
        
        # Set up OpenAI
        openai.api_key = openai_api_key
        
        self.pre_filtering_required = pre_filtering_required
        self.retreival_statergy = retreival_statergy

        self.generation_model_name = generation_model_name
        self.generation_temperature = generation_temperature
        self.generation_max_tokens = generate_max_tokens
        self.generation_client = self.initialize_openai_client(openai_api_key)
        self.few_shots_count = few_shots_count

        self.reranking_enabled = reranking_enabled

        PolicyClassifier.__init__(self, openai_api_key, generation_model_name)
        ElasticSearchRetreiver.__init__(self, es_cloud_id, es_api_key, openai_api_key, index_name, embedding_model, splade_model_name, source, rrf_constant, results_size)
        ReRanker.__init__(self, crossencoder_model_name, reranking_window)

        self.generation_required = generation_required

    def initialize_openai_client(self, api_key: str = None) -> openai.OpenAI:
        """
        Initialize OpenAI client with API key.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment.
        
        Returns:
            OpenAI: Configured OpenAI client
        """
        # Use provided API key or get from environment
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided either as an argument or "
                "through the OPENAI_API_KEY environment variable"
            )
        
        return openai.OpenAI(api_key=api_key)

    def generate_response(
            self,
            user_query: str, 
            context_examples: List[Dict[str, str]],
        ) -> Dict[str, str]:
        """
        Handles context examples with metadata and returns response with citations.
        
        Args:
            user_query (str): The user's question
            context_examples (List[Dict]): List of dicts containing text and metadata
            model (str): OpenAI model to use
        
        Returns:
            Dict containing response text and citation information
        """

        # Prepare context with metadata
        if self.generation_required:
            formatted_contexts = []
            for i, example in enumerate(context_examples):
                context_text = example.get('chunks', '')
                source = example.get('source', 'Unknown')
                page = example.get('page', 'N/A')
                formatted_context = f"[{i+1}] Source: {source}, Page: {page}\n{context_text}"
                formatted_contexts.append(formatted_context)
            
            combined_context = "\n\n---\n\n".join(formatted_contexts)

            user_message = f"""Context information is below:
            ----------------
            {combined_context}
            ----------------
            Question: {user_query}
            
            Please provide a detailed answer with citations."""

            try:
                response = self.generation_client.chat.completions.create(
                    model=self.generation_model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.generation_temperature,
                    max_tokens=self.generation_max_tokens
                )
                
                return {
                    "response": response.choices[0].message.content.strip(),
                    "sources": context_examples
                }
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                raise Exception(error_msg)
        
        else:
            return {
                    "response": "Generation condition triggered off",
                    "sources": context_examples
                }
        
    def get_answer(self, user_query):
        """
        Accepts the user_query and returns the response along with sources

        Args:
            user_query (str): The user's question

        Returns:
            Dict containing response text and citation information
        """
        if self.pre_filtering_required:
            policy_filter = self.get_policy_tag(user_query)
        else:
            policy_filter = 'unknown'
        context = self.search(user_query, policy_filter, self.retreival_statergy)
        if self.reranking_enabled:
            context = self.re_ranker(query=user_query, results=context)
        contexts_with_metadata = [{
            "chunks": x['_source']['content'],
            "_id": x['_id'],
            "_score": str(x['_score']),
            "source": x['_source']['metadata']['pdf_source'],
            "page": "1"
        } for x in context['hits']['hits'][:self.few_shots_count]]
        response = self.generate_response(user_query, contexts_with_metadata)
        return response
    
if __name__ == "__main__":
    bot = QABot(
            es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
            es_api_key=os.getenv("ELASTIC_API_KEY"),
            openai_api_key=os.getenv("openai_api_key"),
            index_name=os.getenv("index_name"),
            retreival_statergy="dense"
    )
    response = bot.get_answer("What is the policy on sick leaves?")
    print('RESPONSE: ', response['response'])
    print('SOURCES: ', response['sources'])