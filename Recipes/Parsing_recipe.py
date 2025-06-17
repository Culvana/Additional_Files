from typing import List, Dict, Any
import logging
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import sys
from datetime import datetime
import json

def setup_logging(log_file: str = "mariachi_indexer.log") -> logging.Logger:
    logger = logging.getLogger("mariachi_indexer")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config() -> Dict[str, Any]:
    load_dotenv()
    required_vars = ['AZURE_AISEARCH_ENDPOINT', 'AZURE_AISEARCH_APIKEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        'search_endpoint': os.getenv('AZURE_AISEARCH_ENDPOINT'),
        'search_key': os.getenv('AZURE_AISEARCH_APIKEY'),
        'openai_key': os.getenv('OPENAI_API_KEY'),
        'index_name': 'mariachi-bakery',
        'embedding_model': 'text-embedding-ada-002',
        'vector_dimensions': 1536,
        'batch_size': 10,
        'max_workers': 4
    }

def get_search_clients(config: Dict[str, Any]) -> tuple[SearchIndexClient, SearchClient]:
    credential = AzureKeyCredential(config['search_key'])
    index_client = SearchIndexClient(endpoint=config['search_endpoint'], credential=credential)
    search_client = SearchClient(
        endpoint=config['search_endpoint'],
        credential=credential,
        index_name=config['index_name']
    )
    return index_client, search_client

def setup_search_index(index_client: SearchIndexClient, config: Dict[str, Any], logger: logging.Logger) -> None:
    try:
        try:
            index_client.delete_index(config['index_name'])
            logger.info("Existing index deleted")
        except Exception as e:
            logger.info(f"No existing index to delete: {e}")

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

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="supplier_name", type=SearchFieldDataType.String),
            SearchableField(name="inventory_item", type=SearchFieldDataType.String),
            SearchableField(name="brand", type=SearchFieldDataType.String),
            SimpleField(name="inventory_unit", type=SearchFieldDataType.String),
            SearchableField(name="item_name", type=SearchFieldDataType.String),
            SimpleField(name="item_number", type=SearchFieldDataType.String),
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
                vector_search_dimensions=config['vector_dimensions'],
                vector_search_profile_name="mariachi-vector-profile"
            )
        ]

        index = SearchIndex(name=config['index_name'], fields=fields, vector_search=vector_search)
        index_client.create_index(index)
        logger.info("Search index created successfully")

    except Exception as e:
        logger.error(f"Error setting up search index: {e}")
        raise

def generate_embedding(text: str, openai_client: OpenAI, config: Dict[str, Any]) -> List[float]:
    response = openai_client.embeddings.create(
        model=config['embedding_model'],
        input=text
    )
    return [float(x) for x in response.data[0].embedding]

def clean_numeric(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def clean_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', 't', 'yes', 'y', '1')
    return bool(value)

def process_inventory_item(index: int, item: Dict[str, Any], openai_client: OpenAI, config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    try:
        # Generate sequential ID
        doc_id = str(index + 1)  # Convert to string since Azure Search expects string keys
        
        # Clean and process inventory item
        inventory_item = clean_string(item["Inventory Item Name"])
        if not inventory_item:
            logger.error(f"Empty inventory item name for index: {doc_id}")
            return None
            
        embedding = generate_embedding(inventory_item, openai_client, config)
        
        document = {
            "id": doc_id,  # Simple numeric ID
            "supplier_name": clean_string(item["Supplier Name"]),
            "inventory_item": inventory_item,
            "brand": clean_string(item["Brand"]),
            "inventory_unit": clean_string(item["Inventory Unit of Measure"]),
            "item_name": clean_string(item["Item Name"]),
            "item_number": clean_string(item["Item Number"]),
            "quantity_in_case": clean_numeric(item["Quantity In Case"]),
            "measurement_of_each_item": clean_numeric(item["Measurement Of Each Item"]),
            "measured_in": clean_string(item["Measured In"]),
            "total_units": clean_numeric(item["Total Units"]),
            "case_price": clean_numeric(item["Case Price"]),
            "catch_weight": clean_boolean(item["Catch Weight"]),
            "priced_by": clean_string(item["Priced By"]),
            "splitable": clean_boolean(item["Splitable"]),
            "split_price": clean_numeric(item["Split Price"]),
            "cost_per_unit": clean_numeric(item["Cost of a Unit"]),
            "source_file": clean_string(item["Source File"]),
            "last_updated": datetime.utcnow().isoformat(),
            "active": clean_boolean(item["Active"]),
            "inventory_item_vector": embedding
        }

        # Verify JSON serialization
        try:
            json.dumps(document, allow_nan=False)
            return document
        except Exception as e:
            logger.error(f"JSON serialization failed for ID {doc_id}: {e}")
            return None

    except Exception as e:
        logger.error(f"Error processing item with index {index}: {e}")
        return None

def validate_document(doc: Dict[str, Any], logger: logging.Logger) -> bool:
    try:
        # Validate required fields
        if not doc.get('id') or not doc.get('inventory_item'):
            return False
            
        # Validate vector field
        vector = doc.get('inventory_item_vector', [])
        if not isinstance(vector, list) or len(vector) != 1536:
            return False
            
        # Test JSON serialization
        json.dumps(doc, allow_nan=False)
        return True
        
    except Exception as e:
        logger.error(f"Document validation failed: {e}")
        return False

def upload_batch(batch: List[Dict[str, Any]], search_client: SearchClient, logger: logging.Logger) -> bool:
    if not batch:
        return True
        
    try:
        valid_docs = [doc for doc in batch if validate_document(doc, logger)]
        if not valid_docs:
            return False
            
        result = search_client.upload_documents(documents=valid_docs)
        return all(not hasattr(item, 'error') for item in result)
        
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        return False

def index_inventory(inventory_data: List[Dict[str, Any]], config: Dict[str, Any], 
                   search_client: SearchClient, openai_client: OpenAI, logger: logging.Logger) -> None:
    processed_documents = []
    successful_items = 0
    failed_items = 0

    # Process items with sequential IDs
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = {}
        for idx, item in enumerate(inventory_data):
            future = executor.submit(process_inventory_item, idx, item, openai_client, config, logger)
            futures[future] = idx

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            try:
                doc = future.result()
                if doc and validate_document(doc, logger):
                    processed_documents.append(doc)
                    successful_items += 1
                else:
                    failed_items += 1
            except Exception as e:
                logger.error(f"Processing error for index {futures[future]}: {e}")
                failed_items += 1

    logger.info(f"Processing complete. Successful: {successful_items}, Failed: {failed_items}")

    if not processed_documents:
        logger.error("No valid documents to upload")
        return

    # Sort documents by ID to maintain order
    processed_documents.sort(key=lambda x: int(x['id']))

    # Upload in batches
    batch_size = config['batch_size']
    successful_uploads = 0
    failed_uploads = 0

    for i in range(0, len(processed_documents), batch_size):
        batch = processed_documents[i:i + batch_size]
        if upload_batch(batch, search_client, logger):
            successful_uploads += len(batch)
        else:
            # If batch fails, try one by one
            for doc in batch:
                if upload_batch([doc], search_client, logger):
                    successful_uploads += 1
                else:
                    failed_uploads += 1

    logger.info(f"Upload complete. Successful: {successful_uploads}, Failed: {failed_uploads}")

def main():
    try:
        logger = setup_logging()
        config = load_config()
        index_client, search_client = get_search_clients(config)
        openai_client = OpenAI(api_key=config['openai_key'])

        inventory_file = Path(r"C:\Users\rahul\OneDrive\Desktop\Recipes\consolidated_inventory_output.csv")
        if not inventory_file.exists():
            raise FileNotFoundError("Inventory file not found")
        
        inventory_data = pd.read_csv(inventory_file).to_dict('records')
        setup_search_index(index_client, config, logger)
        index_inventory(inventory_data, config, search_client, openai_client, logger)
        
        logger.info("Indexing process completed")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    main()