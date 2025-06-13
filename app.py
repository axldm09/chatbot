from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from qdrant_client import QdrantClient
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate

def create_rag_chatbot(debug_mode=False):
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large",
        embed_batch_size=10
    )
    llm = Ollama(model="llama3.2:latest", request_timeout=120.0) 
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="pdf_documents")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")

    index = load_index_from_storage(storage_context=storage_context)
    
    # Better reranker with higher score
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=5
    )
    
    # Improved prompt with focus on specificity
    scientific_prompt = PromptTemplate(
        """Eres un asistente experto en investigación científica. Tu tarea es responder preguntas utilizando ÚNICAMENTE la información proporcionada en el contexto. No inventes información.
        
        IMPORTANTE: Responde a la pregunta con datos específicos del contexto pero con un tono fluido y conversacional. EVITA respuestas genéricas.
        
        - Menciona a los autores por sus nombres cuando sea relevante (ej: "Salvador Pane ha investigado sobre...")
        - Estructura la respuesta de forma natural, como lo haría un experto explicando el tema
        - Mantén un estilo informativo pero accesible
        
        Contexto: {context}
        Pregunta: {query}
        
        Respuesta (incluye datos específicos con un tono fluido y natural):"""
    )
    
    query_engine = index.as_query_engine(
        text_qa_template=scientific_prompt,
        node_postprocessors=[reranker],
        similarity_top_k=10,  # Increased for better recall
    )

    return query_engine, index

def main():
    try:
        debug_mode = False   # Set to True to see retrieved chunks
        query_engine, index = create_rag_chatbot(debug_mode)
        
        print("Chatbot RAG de artículos científicos inicializado. Escribe 'salir' para terminar. Escribe 'debug' para activar/desactivar el modo depuración.")
        while True:
            user_input = input("\nPregunta: ")
            if user_input.lower() == 'salir':
                break
            elif user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"Modo depuración: {'activado' if debug_mode else 'desactivado'}")
                query_engine, index = create_rag_chatbot(debug_mode)
                continue
            
            try:
                print("\nBuscando información relevante...")
                response = query_engine.query(user_input)
                
                # Better handling of empty or generic responses
                if len(response.response.strip()) < 30 or "no tengo suficiente información" in response.response.lower():
                    # Try with more aggressive retrieval as fallback
                    fallback_engine = index.as_query_engine(
                        similarity_top_k=20,
                        response_mode="compact"
                    )
                    fallback_response = fallback_engine.query(user_input)
                    print(f"\nRespuesta: {fallback_response.response}")
                else:
                    print(f"\nRespuesta: {response.response}")
                    
                # Show source nodes in debug mode
                if debug_mode and hasattr(response, 'source_nodes'):
                    print("\n--- Fuentes utilizadas ---")
                    for i, node in enumerate(response.source_nodes):
                        print(f"Fuente {i+1} (Score: {node.score:.4f}):\n{node.text[:150]}...\n")
                    
            except Exception as e:
                print(f"\nError al procesar la consulta: {str(e)}")
                
    except Exception as e:
        print(f"Error al inicializar el chatbot: {str(e)}")

if __name__ == "__main__":
    main()
