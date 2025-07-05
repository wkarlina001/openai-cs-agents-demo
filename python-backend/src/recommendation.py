import json
import os
import chromadb
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging #

import yaml
def load_config_from_yaml(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        return config if config is not None else {} # Return empty dict if YAML is empty

config_filepath = 'config/config.yaml'
app_config = load_config_from_yaml(config_filepath)

CHROMA_RECO_PATH = app_config.get('CHROMA_RECO_PATH')
RECO_COLLECTION_NAME = app_config.get('RECO_COLLECTION_NAME')
EMBEDDING_MODEL_NAME = app_config.get('EMBEDDING_MODEL_NAME')
TELCO_PLAN_DOCUMENT_PATH = app_config.get('TELCO_PLAN_DOCUMENT_PATH')
MY_MODEL = app_config.get('MY_MODEL')
LLM_TEMPERATURE = app_config.get('LLM_TEMPERATURE')
RECO_TOP_N = app_config.get('RECO_TOP_N')

logger = logging.getLogger(__name__) #

class CustomRecommendationSystem:
    """
    A placeholder for a custom recommendation system.
    """
    def __init__(self,
                 model : str = MY_MODEL, #
                 document_path : str = TELCO_PLAN_DOCUMENT_PATH, #
                 chroma_path : str = CHROMA_RECO_PATH, #
                 collection_name: str = RECO_COLLECTION_NAME, #
                 embedding_model_name: str = EMBEDDING_MODEL_NAME, #
                 llm_temperature: float = LLM_TEMPERATURE #
                ):
        logger.info("CustomRecommendationSystem initialized.")
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        ) 

        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_function)
        logger.info(f"ChromaDB initialized at: {chroma_path}")
        logger.info(f"Collection '{self.collection_name}' ready. Current items: {self.collection.count()}")

        # Add some initial products if the collection is empty
        if self.collection.count() == 0:
            logger.info("ChromaDB collection is empty. Adding initial example products...")
            self.products_data = self._load_mock_data(document_path) # Mock data for demonstration
            self._add_products()

        self.llm = ChatOpenAI(
            temperature=llm_temperature,
            model_name=model,
            max_tokens=300,
        )

    def _load_mock_data(self, mock_file):
        """
        Loads product data from a mock JSON file and groups it by category.

        Args:
            mock_file (str): The path to the mock JSON file.

        Returns:
            dict: A dictionary where keys are product categories and values are
                  lists of products belonging to that category. Returns an empty
                  dictionary if the file is not found, or if there's a JSON
                  decoding error or an invalid structure.
        """
        if not os.path.exists(mock_file):
            logger.info(f"Error: File not found at {mock_file}")
            return {} # Return an empty dictionary for consistent type

        try:
            with open(mock_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, dict): 
                    return data
                else:
                    logger.error(f"Error: JSON structure invalid. Expected a dict of products.")
                    return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {mock_file}: {e}")
            return {}
        except Exception as e:
            logger.exception(f"An unexpected error occurred while reading {mock_file}: {e}")
            return {}
        
    def _add_products(self):
        # Prepare Documents, Metadata, and IDs for ChromaDB 
        documents = []
        metadatas = []
        ids = []
        product_id_counter = 0

        for category, products_list in self.products_data.items():
            for product in products_list:
                doc_content = f"Product Name: {product['productName']}. Description: {product['productDescription']}"
                documents.append(doc_content)

                # Ensure metadata values are simple types (str, int, float, bool)
                metadata = {
                    "category": str(category),
                    "product_name": str(product["productName"]),
                    "product_description": str(product["productDescription"]),
                    "price_per_month_sgd": float(product["price_per_month_sgd"]),
                    "speed": str(product["speed"]),
                    "contract_period": str(product["contract_period"])
                }
                metadatas.append(metadata)

                # Each document needs a unique ID
                ids.append(f"product_{product_id_counter}")
                product_id_counter += 1

        logger.info(f"Adding {len(documents)} documents to the '{self.collection.name}' collection...")
        try:
            self.collection.add(
                documents=documents, # Pass documents; Chroma generates embeddings automatically
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Documents added successfully!")
            logger.info(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}") #


    def get_custom_recommendations(self, query: str = None, top_n: int = RECO_TOP_N) -> list[dict]:
        """
        Finds products in the database that are semantically similar to the given query.
        Chroma will automatically generate the query embedding using the configured embedding_function.

        Args:
            query (str): The user's query (e.g., "tech gadgets for productivity",
                         "something to listen to music with").
            top_n (int): The number of top similar products to return.

        Returns:
            list[dict]: A list of dictionaries, each containing 'name', 'description'.
        """
        if not query:
            logger.warning("Query is empty. Returning error message.") #
            return [{"name": "Error", "description": "Provide non-empty query."}]

        try:
            # Pass the query text directly to Chroma; it will embed it using the configured EF
            # Include 'embeddings' in the return to get the vectors
            results = self.collection.query(
                    query_texts=[query], # Use query_texts for automatic embedding
                    n_results=top_n,
                    include=['metadatas']
                )
            
            recommended_products = []
            if results and results['metadatas']:
                for i in range(len(results['metadatas'][0])):
                    metadata = results['metadatas'][0][i]
                    if metadata:
                        recommended_products.append({
                            "name": metadata.get("product_name", "Unknown Product"),
                            "description": metadata.get("product_description", "No description available.")
                        })
            
            if not recommended_products:
                return [{"name": "Error", "description": "No suitable product."}]
            else:
                return recommended_products

        except Exception as e:
            logger.exception(f"Error querying ChromaDB for recommendations: {e}") #
            return [{"name": "Error", "description": "An error occurred during recommendation search."}]
    
    def generate_answer(self, query):
        anws = self.get_custom_recommendations(query)
        
        if anws:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Based on the recommendation options, remove the options that are not relevant to the query.
                Reorder the options from the most to least relevant.
                Return all of the remaining recommendation options with product name followed by brief description.
                If none of the options are relevant, reply - Sorry, I can't find suitable products.

                Context:
                {context}

                Question: {question}

                Answer:"""
            )

            try:
                response = self.llm.invoke(prompt.format(context=anws, question=query))
                logger.info(f"Generated recommendation answer: {response.content[:200]}...") #
                return response.content
            except Exception as e:
                logger.exception(f"Error generating LLM answer for recommendations: {e}") #
                return "Sorry, I encountered an issue while generating recommendations."
        else:
            logger.warning("No recommendations retrieved from ChromaDB, cannot generate answer.") #
            return "Sorry, I couldn't find any recommendations based on your request."

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) #
    reco_sys = CustomRecommendationSystem()

    query = "recommend prepaid mobile plan with cheap price and no contract"
    answer = reco_sys.generate_answer(query)
    print(f"\nQuery: {query}\nAnswer: {answer}")
    print("-" * 50)

    query = "i want something for gaming"
    answer = reco_sys.generate_answer(query)
    print(f"\nQuery: {query}\nAnswer: {answer}")
    print("-" * 50)

    query = "what is the best broadband plan?"
    answer = reco_sys.generate_answer(query)
    print(f"\nQuery: {query}\nAnswer: {answer}")
    print("-" * 50)

    query = "tell me about your services" # Irrelevant query
    answer = reco_sys.generate_answer(query)
    print(f"\nQuery: {query}\nAnswer: {answer}")