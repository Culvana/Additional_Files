from typing import List, Dict, Optional
from dataclasses import dataclass
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
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_id_string(text: str) -> str:
    """Clean string to make it suitable for use as an ID"""
    if pd.isna(text):
        return "none"
    # Remove invalid characters and replace spaces/slashes with underscores
    cleaned = ''.join(c for c in str(text) if c.isalnum() or c in '_ -/').strip()
    cleaned = cleaned.replace(' ', '_').replace('/', '_').replace('-', '_')
    return cleaned.lower()  # Convert to lowercase for consistency

@dataclass
class Config:
    search_endpoint: str
    search_key: str
    openai_key: str

    @classmethod
    def from_env(cls) -> 'Config':
        load_dotenv()
        return cls(
            search_endpoint=os.getenv('AZURE_AISEARCH_ENDPOINT', ''),
            search_key=os.getenv('AZURE_AISEARCH_APIKEY', ''),
            openai_key=os.getenv('OPENAI_API_KEY', '')
        )

class MariachiBakerySearch:
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.openai_key)
        self.index_client = SearchIndexClient(
            endpoint=config.search_endpoint,
            credential=AzureKeyCredential(config.search_key)
        )
        self.search_client = None  # Will be initialized after index creation

    def setup_search_index(self) -> None:
        """Create or recreate Azure Search index with vector search capabilities"""
        try:
            # Delete the existing index if it exists
            try:
                self.index_client.delete_index("mariachi-bakery")
                logger.info("Existing index deleted successfully")
            except Exception as e:
                logger.info(f"No existing index to delete or error deleting: {e}")

            # Define vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="mariachi-vector-config",
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
                        name="mariachi-vector-profile",
                        algorithm_configuration_name="mariachi-vector-config"
                    )
                ]
            )

            # Define fields
            fields = [
                SimpleField(name="unique_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="item_number", type=SearchFieldDataType.String),
                SearchableField(name="supplier_name", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
                SearchableField(name="inventory_item", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
                SearchableField(name="brand", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
                SimpleField(name="inventory_unit", type=SearchFieldDataType.String),
                SearchableField(name="item_name", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
                SimpleField(name="quantity_in_case", type=SearchFieldDataType.Double),
                SimpleField(name="measurement_of_each_item", type=SearchFieldDataType.Double),
                SimpleField(name="measured_in", type=SearchFieldDataType.String),
                SimpleField(name="total_units", type=SearchFieldDataType.Double),
                SimpleField(name="case_price", type=SearchFieldDataType.Double),
                SimpleField(name="catch_weight", type=SearchFieldDataType.Boolean),
                SimpleField(name="priced_by", type=SearchFieldDataType.String),
                SimpleField(name="splitable", type=SearchFieldDataType.Boolean),
                SimpleField(name="split_price", type=SearchFieldDataType.Double),
                SimpleField(name="cost_per_unit", type=SearchFieldDataType.Double),
                SimpleField(name="source_file", type=SearchFieldDataType.String),
                SimpleField(name="last_updated", type=SearchFieldDataType.String),
                SimpleField(name="active", type=SearchFieldDataType.Boolean),
                SearchField(
                    name="inventory_item_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="mariachi-vector-profile"
                )
            ]

            # Create new index
            index = SearchIndex(
                name="mariachi-bakery",
                fields=fields,
                vector_search=vector_search
            )

            self.index_client.create_index(index)
            logger.info("Search index created successfully")
            
            # Initialize the search client after index creation
            self.search_client = SearchClient(
                endpoint=self.config.search_endpoint,
                credential=AzureKeyCredential(self.config.search_key),
                index_name="mariachi-bakery"
            )

        except Exception as e:
            logger.error(f"Error setting up search index: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API with error handling"""
        try:
            if pd.isna(text) or not text:
                logger.warning("Empty or NaN text provided for embedding")
                return None
                
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=str(text).strip()
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {text}. Error: {e}")
            raise

    def process_inventory_item(self, item: Dict) -> Optional[Dict]:
        torch.manual_seed(10000)
        """Process a single inventory item"""
        try:
            unique_id = torch.randint(0,100000,(1,))
            
            # Generate embedding for the inventory item name
            inventory_item_name = str(item["Inventory Item Name"]).strip()
            embedding = self.generate_embedding(inventory_item_name)
            
            if not embedding:
                logger.warning(f"Could not generate embedding for item: {unique_id}")
                return None

            # Create the document
            return {
                "unique_id": unique_id,
                "item_number": str(item["Item Number"]),
                "supplier_name": str(item["Supplier Name"]),
                "inventory_item": inventory_item_name,
                "brand": str(item["Brand"]),
                "inventory_unit": str(item["Inventory Unit of Measure"]),
                "item_name": str(item["Item Name"]),
                "quantity_in_case": float(item["Quantity In Case"]) if pd.notna(item["Quantity In Case"]) else 0.0,
                "measurement_of_each_item": float(item["Measurement Of Each Item"]) if pd.notna(item["Measurement Of Each Item"]) else 0.0,
                "measured_in": str(item["Measured In"]),
                "total_units": float(item["Total Units"]) if pd.notna(item["Total Units"]) else 0.0,
                "case_price": float(item["Case Price"]) if pd.notna(item["Case Price"]) else 0.0,
                "catch_weight": bool(item["Catch Weight"]) if pd.notna(item["Catch Weight"]) else False,
                "priced_by": str(item["Priced By"]),
                "splitable": bool(item["Splitable"]) if pd.notna(item["Splitable"]) else False,
                "split_price": float(item["Split Price"]) if pd.notna(item["Split Price"]) else 0.0,
                "cost_per_unit": float(item["Cost of a Unit"]) if pd.notna(item["Cost of a Unit"]) else 0.0,
                "source_file": str(item["Source File"]),
                "last_updated": str(item["Last Updated At"]),
                "active": bool(item["Active"]) if pd.notna(item["Active"]) else False,
                "inventory_item_vector": embedding
            }
        except Exception as e:
            logger.error(f"Error processing inventory item: {e}")
            return None

    def index_inventory(self, inventory_data: List[Dict]) -> None:
        """Index inventory items with parallel processing"""
        documents = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_inventory_item, item) 
                for item in inventory_data
            ]
            
            for future in tqdm(futures, desc="Processing inventory items"):
                try:
                    result = future.result()
                    if result:
                        documents.append(result)
                except Exception as e:
                    logger.error(f"Error processing inventory item: {e}")

        if not documents:
            logger.error("No valid documents to index")
            return

        # Upload in optimized batches
        batch_size = 50  # Reduced batch size for better reliability
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                self.search_client.upload_documents(documents=batch)
                logger.info(f"Indexed batch of {len(batch)} documents")
            except Exception as e:
                logger.error(f"Error uploading batch: {e}")

def main():
    try:
        # Initialize configuration
        config = Config.from_env()
        search_service = MariachiBakerySearch(config)
        
        # Set up file paths
        inventory_file = Path(r"C:\Users\rahul\OneDrive\Desktop\Recipes\consolidated_inventory_output.csv")
        
        if not inventory_file.exists():
            raise FileNotFoundError(f"Inventory file not found at {inventory_file}")
        
        # Read and process inventory data
        try:
            inventory_data = pd.read_csv(inventory_file).to_dict('records')
            logger.info(f"Successfully loaded {len(inventory_data)} inventory records")
        except Exception as e:
            logger.error(f"Error reading inventory file: {e}")
            raise
        
        # Create search index
        search_service.setup_search_index()
        
        # Index inventory items
        search_service.index_inventory(inventory_data)
        
        logger.info("Successfully completed indexing inventory items")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    main()