from openai import OpenAI
from dotenv import load_dotenv
import os

try:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        client = OpenAI(api_key=api_key)
        system_prompt = "You are a helpful assistant."
        
        response = client.chat.completions.create(
            model='gpt-4',  
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': 'What is the purpose of life?'}
            ]
        )
        print(response.choices[0].message.content)
        ### Embeddings 
        embeddings=client.embeddings.create(input="Hello, world!",model="text-embedding-ada-002")
        print(embeddings.data[0].embedding)
        ### Finetuning Need JSonl file
        client.files.create(file=open("file.jsonl", "rb"),purpose="fine-tune")
        ###
        index_name="my_index"
        def embedding(text: str):
            response = client.embeddings.create(input=text, model="text-embedding-ada-002")
            return response.data[0].embedding
        def set_index():
            fields=[simpleField(name="id",type=SearchFieldDataType.STRING),SearchField(name="",type=SearchFieldDataType.STRING)]



except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Error occurred: {e}")



         