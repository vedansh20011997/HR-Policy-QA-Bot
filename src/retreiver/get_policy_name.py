import os
from dotenv import load_dotenv
load_dotenv()
from typing import Literal, Optional, List
import openai
import json
from datetime import datetime

class PolicyClassifier:
    """
    A class to classify natural language queries into predefined policy categories using OpenAI's API.
    """
    
    PolicyType = Literal[
        "leave policy",
        "grievance and disciplinary policy",
        "travel reimbursement policy",
        "posh policy",
        "esops policy",
        "information security and it policy",
        "parental leave policy",
        "remote work policy",
        "expense reimbursement policy",
        "recruitment and onboarding policy",
        "unknown"
    ]

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the PolicyClassifier with OpenAI API key and model.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI model to use (default: gpt-4-turbo-preview)
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.model = model
        self._setup_functions()
        self.classification_history = []

    def _setup_functions(self) -> None:
        """Set up the function definition for OpenAI function calling."""
        self.functions = [
            {
                "name": "classify_policy",
                "description": "Classify a policy related query into the most relevant policy category",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "policy_type": {
                            "type": "string",
                            "enum": [
                                "leave policy",
                                "grievance and disciplinary policy",
                                "travel reimbursement policy",
                                "posh policy",
                                "esops policy",
                                "information security and it policy",
                                "parental leave policy",
                                "remote work policy",
                                "expense reimbursement policy",
                                "recruitment and onboarding policy",
                                "unknown"
                            ],
                            "description": "The type of policy that best matches the query"
                        }
                    },
                    "required": ["policy_type"]
                }
            }
        ]

    def get_policy_tag(self, query: str) -> Optional[PolicyType]:
        """
        Classify a natural language query into a policy category.
        
        Args:
            query (str): Natural language query about a policy
            
        Returns:
            Optional[PolicyType]: The matched policy category or None if classification fails
        """
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a policy classification assistant. Your task is to categorize policy-related queries 
                                    into the correct policy type. If the question is out of context wrt mentioned policy types, return unknown.
                                    Consider the context and intent of the query carefully."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                functions=self.functions,
                function_call={"name": "classify_policy"}
            )

            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "classify_policy":
                policy_type = json.loads(function_call.arguments)["policy_type"]
                
                # Store classification history
                self.classification_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "classification": policy_type
                })
                
                return policy_type

            return None

        except Exception as e:
            self._log_error(f"Error classifying policy: {str(e)}")
            return None

    def get_classification_history(self) -> List[dict]:
        """
        Get the history of all classifications made.
        
        Returns:
            List[dict]: List of classification records with timestamps
        """
        return self.classification_history

    def get_available_policies(self) -> List[str]:
        """
        Get a list of all available policy categories.
        
        Returns:
            List[str]: List of policy categories
        """
        return [
            "leave policy",
            "grievance and disciplinary policy",
            "travel reimbursement policy",
            "posh policy",
            "esops policy",
            "information security and it policy",
            "parental leave policy",
            "remote work policy",
            "expense reimbursement policy",
            "recruitment and onboarding policy"
        ]

    def _log_error(self, error_message: str) -> None:
        """
        Log error messages. Can be extended to use proper logging framework.
        
        Args:
            error_message (str): Error message to log
        """
        print(f"[ERROR] {datetime.now().isoformat()}: {error_message}")


# Example usage
if __name__ == "__main__":
    # Initialize the classifier
    classifier = PolicyClassifier(api_key=os.getenv("openai_api_key"))
    
    # Test queries
    test_queries = [
        "How many personal leaves are there?",
        "Explain about the sexual harassment policy?",
        "What is the process for claiming travel expenses?",
        "How do stock options vest?",
        "What are the guidelines for working from home?",
        "Which is the highest mountain peak?"
    ]
    
    # Run test classifications
    for query in test_queries:
        result = classifier.get_policy_tag(query)
        print(f"\nQuery: {query}")
        print(f"Classified as: {result}")
    
    # Get classification history
    print("\nClassification History:")
    for record in classifier.get_classification_history():
        print(f"Time: {record['timestamp']}")
        print(f"Query: {record['query']}")
        print(f"Classification: {record['classification']}\n")