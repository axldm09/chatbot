import os
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

def pass_documents_to_qdrant():
    # 1. Load documents using LlamaIndex PDF loader
    pdf_reader = PDFReader()
    documents = []
    data_dir = "./data"
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        documents.extend(pdf_reader.load_data(file=pdf_file))
    
    # 2. Split documents into manageable chunks
    text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=10)
    nodes = text_splitter.get_nodes_from_documents(documents)
    
    # 3. Configure settings
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large", embed_batch_size=10, show_progress_bar=True)
    Settings.embed_model = embed_model
    
    # 4. Create vector store and index
    client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="pdf_documents")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Indexing {len(nodes)} document chunks...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context
    )

    # Save the index metadata
    index.storage_context.persist(persist_dir="./storage")
    print("Indexing complete.")
    client.close()

def search_similar_documents(query_text, top_k=5):
    # 1. Load the existing index
    client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="pdf_documents")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store)
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large", embed_batch_size=10)
    Settings.llm = None  
    
    # 2. Create query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    
    # 3. Perform the search
    response = query_engine.query(query_text)
    
    # 4. Display results with similarity scores
    print(f"Query: {query_text}")
    print("Similar documents:")
    
    for node in response.source_nodes:
        print(f"Score: {node.score:.4f}")
        print(f"Content: {node.node.get_content()[:150]}...")

if __name__=="__main__":
    pass_documents_to_qdrant()
    # Example search
    search_similar_documents("¿Quien articulos ha escrito Salvador Pane? ", top_k=5)
    print("Documents passed to Qdrant successfully.")