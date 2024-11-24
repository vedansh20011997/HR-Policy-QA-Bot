from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
load_dotenv()
import json
from tqdm import tqdm
import os
import sys
sys.path.append('../../')
from src.retreiver.es_retreiver import ElasticSearchRetreiver

index = ElasticSearchRetreiver(
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        es_api_key=os.getenv("ELASTIC_API_KEY"),
        openai_api_key=os.getenv("openai_api_key"),
        index_name=os.getenv("index_name"))

class EvaluationDatasetGenerator:
    def __init__(self, openai_api_key):
        # Set up OpenAI
        openai.api_key = openai_api_key
        self.generation_client = self.initialize_openai_client(openai_api_key)
    
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
    
    def generate_questions_gpt(self, chunk: str, num_questions: int = 3) -> List[Dict[str, str]]:
        """Generate questions using GPT-4"""
        prompt = f"""Given the following text passage, generate {num_questions} diverse questions that can be answered using this passage. 
        For each question, provide:
        A question that tests understanding of the content

        Passage: {chunk}

        Return question string in response
        """
        
        response = self.generation_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        question_string = response.choices[0].message.content
        return question_string
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: str):
        """Save evaluation dataset to file"""
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

    def load_dataset(self, input_path: str) -> Dict[str, Any]:
        """Load evaluation dataset from file"""
        with open(input_path, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    generator = EvaluationDatasetGenerator(os.getenv('openai_api_key'))
    questions_to_generate = 100
    ground_truth_dump = []

    while(questions_to_generate):
        dict_to_write = dict()
        random_document = index.es.search(
             index=os.getenv("index_name"),
             query={
                  "function_score": {
                       "query": {"match_all": {}},
                       "random_score": {},
                       "boost_mode": "replace"
                       }
                       },
                       source = ['content'],
                       size=1)
        dict_to_write['_id'] = random_document['hits']['hits'][0]["_id"]
        dict_to_write['content'] = random_document['hits']['hits'][0]["_source"]["content"]

        # Create evaluation dataset
        question = generator.generate_questions_gpt(
            chunk=dict_to_write['content'],
            num_questions=1
        )
        dict_to_write["question"] = question
        ground_truth_dump.append(dict_to_write)
        questions_to_generate -= 1

        print(f'Questions left to generate: {questions_to_generate}')
        with open('ground_truth_data.json', 'w') as file:
            json.dump(ground_truth_dump , file)