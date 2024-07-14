import json
import pandas as pd 

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from langchain.vectorstores import Qdrant
import json

# generate log
import logging
from pathlib import Path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Load the CSV file into a pandas DataFrame
df = pd.read_csv("/home/bammwise/python-development/ragvectors/sampleLLama3RaG/Dataset/financial_news.csv")
print(df.head())


# define model for save from  sentence-transformers/all-MiniLM-L12-v2

# def download_model_and_tokenizer(model_name, save_path):
#     """
#     Download and save both the model and the tokenizer to the specified directory.

#     Parameters:
#         model_name (str): Name of the model to download.
#         save_path (str or Path): Path to the directory where the model and tokenizer will be saved.
#     """
#     # Create the save path if it doesn't exist
#     save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)
    
#     # Initialize tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
    
#     # Save tokenizer
#     tokenizer.save_pretrained(save_path)
    
#     # Save model
#     model.save_pretrained(save_path)

# # Example usage
# model_name = 'sentence-transformers/all-MiniLM-L12-v2'  # Model name to download
# save_path = Path("/home/bammwise/python-development/ragvectors/sampleLLama3RaG/MiniLM-L12-v2/")  # Path where model and tokenizer will be saved
# download_model_and_tokenizer(model_name, save_path)



# load model for local address from  sentence-transformers/all-MiniLM-L12-v2
def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified directory.

    Parameters:
        model_path (str or Path): Path to the directory containing the saved model and tokenizer.

    Returns:
        tokenizer (transformers.PreTrainedTokenizer): Loaded tokenizer.
        model (transformers.PreTrainedModel): Loaded model.
    """
    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model

# Load the model and tokenizer
model_path = Path("/home/bammwise/python-development/ragvectors/sampleLLama3RaG/MiniLM-L12-v2/")
tokenizer, model = load_model_and_tokenizer(model_path)


df['news'] = df.apply(lambda row: str(row['title']) + ' ' + str(row['summary']), axis=1)




import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    # sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sentence_embedding.cpu().numpy().reshape(1, -1)


    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


df['encoded_news'] = df['news'].apply(lambda x: generate_embedding(x)[0].tolist())
print(df.head())



from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

qdrant_client = QdrantClient(host='localhost', port=6333)
collection_name = "persian_voc"

vectors_config = VectorParams(
    size=model.config.hidden_size,  
    distance=Distance.COSINE  
)

# Create or recreate the collection with the specified configuration
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=vectors_config,
)


 # Insert data into Qdrant
for index, row in df.iterrows():
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[{
            "id": index,  
            "vector": row['encoded_news'],
            "payload": {
                # "instruction": row['instruction'], 
                # "input": row['input'],
                "output": row['news']
            }
        }]
    )



from langchain.vectorstores import Qdrant


url = "http://localhost:6333"  
collection_name =  "persian_voc"  

# Initialize the Qdrant client with the specified URL
client = QdrantClient(
    url=url,
    prefer_grpc=False  # Indicates whether to use gRPC for communication
)


# Create a Qdrant object with the specified client, embeddings, and collection name
# Initialize the Qdrant vector store from langchain
db = Qdrant(
    client=client,
    embeddings=df['encoded_news'].tolist(),  # Use the generated embeddings
    collection_name=collection_name
)

logging.info(f"Qdrant vector store initialized: {db}")


# def similarity_search_with_score(query, k=2):
#     query_embedding = generate_embedding(query)[0].tolist()
#     search_results = qdrant_client.search(
#         collection_name=collection_name,
#         query_vector=query_embedding,
#         limit=k,
#         with_payload=True,
#         with_vectors=False
#     )
#     return search_results

# query = "معنی کلمه محمود چی میشه؟"
# search_results = similarity_search_with_score(query=query, k=5)

# for result in search_results:
#     doc_id = result.id
#     score = result.score
#     payload = result.payload  
#     doc_content = payload.get('output', 'No content available')
#     logging.info({"score": score, "doc_id": doc_id, "content": doc_content})




def similarity_search_with_score(query, k=1):
    query_embedding = generate_embedding(query)[0].tolist()
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k,
        with_payload=True,
        with_vectors=False
    )

    results_with_content = []
    for result in search_results:
        doc_id = result.id
        score = result.score
        payload = result.payload  
        doc_content = payload.get('output', 'No content available')
        results_with_content.append((score, doc_content))
    sorted_results = sorted(results_with_content, key=lambda x: x[0], reverse=True)

    # Concatenate the content of the top k results
    concatenated_content = ' '.join([content for _, content in sorted_results[:k]])

    return concatenated_content

query = "what's you job ? "
outlier_paragraph = similarity_search_with_score(query=query, k=1)

# Print the concatenated content
logging.info({"concatenated_content": outlier_paragraph})


logging.info({"query for llm": str(query + outlier_paragraph)[:512] })



from llama_cpp import Llama

llm = Llama(
      model_path="/home/bammwise/ollama/llama-2-7b-chat.Q2_K.gguf",
)
output = llm(
      f"Q: {str(query + outlier_paragraph)[:512]} ? A: ", 
      max_tokens=32, 
      stop=["Q:", "\n"], 
      echo=True 
) 
print(output)










