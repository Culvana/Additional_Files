from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from openai import OpenAI
import logging
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchField
)
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InventoryItem:
    name: str
    unit_price: float
    unit: str
    unit_size: float
    quantity_in_case: int
    case_price: Optional[float]
    is_catch_weight: bool

class RecipeCostAnalyzer:
    def __init__(self, openai_key: str, search_endpoint: str, search_key: str):
        self.openai_client = OpenAI(api_key=openai_key)
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(search_key)
        )
        self.search_client = None
        self.setup_search_index()
        
        # Initialize search client after index is created
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(search_key),
            index_name="recipe-ingredients"
        )
        
        self.unit_conversions = {
            'oz': {'lb': 1/16, 'gram': 28.3495},
            'lb': {'oz': 16, 'gram': 453.592},
            'gram': {'oz': 0.035274, 'lb': 0.00220462},
            'ml': {'fl_oz': 0.033814, 'gallon': 0.000264172},
            'fl_oz': {'ml': 29.5735, 'gallon': 0.0078125},
            'gallon': {'ml': 3785.41, 'fl_oz': 128}
        }

    def setup_search_index(self):
        try:
            # Define vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="recipe-vector-config",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="recipe-vector-profile",
                        algorithm_configuration_name="recipe-vector-config"
                    )
                ]
            )

            # Define fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="name", type=SearchFieldDataType.String),
                SimpleField(name="unit_price", type=SearchFieldDataType.Double),
                SimpleField(name="unit", type=SearchFieldDataType.String),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="recipe-vector-profile"
                )
            ]

            # Create index
            index = SearchIndex(
                name="recipe-ingredients",
                fields=fields,
                vector_search=vector_search
            )

            self.index_client.create_or_update_index(index)
            logger.info("Search index created successfully")

        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            raise

    def index_inventory_items(self, inventory_items: List[InventoryItem]):
        try:
            documents = []
            for i, item in enumerate(inventory_items):
                try:
                    embedding = self.generate_embedding(item.name)
                    doc = {
                        "id": str(i),
                        "name": item.name,
                        "unit_price": item.unit_price if item.unit_price else 0.0,
                        "unit": item.unit,
                        "embedding": embedding
                    }
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error processing item {item.name}: {e}")
                    continue

            if documents:
                # Upload in batches
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    try:
                        self.search_client.upload_documents(documents=batch)
                        logger.info(f"Indexed batch of {len(batch)} documents")
                    except Exception as e:
                        logger.error(f"Error uploading batch: {e}")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for {text}: {e}")
            raise

    def find_best_match(self, ingredient_name: str, inventory_items: List[InventoryItem]) -> Tuple[Optional[InventoryItem], float]:
        try:
            embedding = self.generate_embedding(ingredient_name)
            
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=3,
                fields="embedding"
            )
            
            results = list(self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["name", "unit_price", "unit"]
            ))
            
            if results:
                best_match = next((item for item in inventory_items 
                                 if item.name.lower() == results[0]['name'].lower()), None)
                return best_match, float(results[0]['@search.score'])
            
            return None, 0.0
        except Exception as e:
            logger.error(f"Error finding match for {ingredient_name}: {e}")
            return None, 0.0



def main():
    load_dotenv()
    
    # Load inventory data
    inventory_df = pd.read_csv('consolidated_inventory_output.csv')
    
    try:
        analyzer = RecipeCostAnalyzer(
            openai_key=os.getenv("OPENAI_API_KEY"),
            search_endpoint=os.getenv("AZURE_AISEARCH_ENDPOINT"),
            search_key=os.getenv("AZURE_AISEARCH_APIKEY")
        )
        
        # Process inventory
        inventory_items = analyzer.process_inventory_data(inventory_df)
        
        # Example recipe ingredients
        recipe = [
            {'name': 'Milk, 2% Reduced Fat', 'quantity': 1, 'unit': 'gallon'},
            {'name': 'Sugar, Fine Granulated', 'quantity': 2, 'unit': 'lb'},
            {'name': 'Eggs, Medium White', 'quantity': 12, 'unit': 'each'}
        ]
        
        # Analyze recipe cost
        result = analyzer.analyze_recipe_cost(recipe, inventory_items)
        
        # Print results
        print(f"\nTotal Recipe Cost: ${result['total_cost']:.2f}")
        for cost in result['ingredient_costs']:
            print(f"\n{cost['ingredient']}:")
            print(f"  Quantity: {cost['quantity']} {cost['unit']}")
            print(f"  Matched Item: {cost['matched_item']}")
            print(f"  Unit Cost: ${cost['unit_cost']:.2f}")
            print(f"  Total Cost: ${cost['total_cost']:.2f}")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()