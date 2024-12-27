import chromadb
from langchain_core.documents import Document
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

import streamlit as st

class VectorStore:
    
    def __init__(self, collection_name: str = "rag-app"):
        """Manages vector storage and retrieval operations.
        
        Creates an Ollama embedding function using the nomic-embed-text model 
        
        Attributes:
            collection_name (str): Name of ChromaDB collection
            model_name (str): Name of embedding model
        """
        self.collection_name = collection_name
        self.embedding_function = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
        self.collection = self.get_vector_collection()

    def get_vector_collection(self) ->chromadb.Collection:
        """Gets or creates a ChromaDB collection for vector storage.

        Initializes a persistent ChromaDB client. 
        Returns a collection that can be used to store and query document embeddings.

        Returns:
            chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
                function and cosine similarity space.
        """
        try:
            chroma_client = chromadb.PersistentClient(path="./rag-chroma")
            return chroma_client.get_or_create_collection(
                name="rag-app",
                embedding_function= self.embedding_function,
                metadata= {'hsnw:space': 'cosine'},
            )
        except Exception as e:
            raise (f"Error happened while returning the chroma vector DB collection - {e}")

    def add_to_vector_collection(self, all_splits: list[Document], file_name:str):
        """Adds document splits to a vector collection for semantic search.

        Takes a list of document splits and adds them to a ChromaDB vector collection
        along with their metadata and unique IDs based on the filename.

        Args:
            all_splits: List of Document objects containing text chunks and metadata
            file_name: String identifier used to generate unique IDs for the chunks

        Returns:
            None. Displays a success message via Streamlit when complete.

        Raises:
            ChromaDBError: If there are issues upserting documents to the collection
        """
        try:
            collection = self.collection
            document, metadata, ids = [], [], []

            for idx, split in enumerate(all_splits):
                document.append(split.page_content)
                metadata.append(split.metadata)
                ids.append(f"{file_name}_{idx}")

            collection.upsert(
                documents=document,
                metadatas=metadata,
                ids=ids
            )

            st.success("Data added to the vector store !")
        except Exception as e:
            raise (f"Error happened while upserting documents to the collection - {e}")

    def query_collection(self, prompt:str, n_results: int=10):
        """Query vector store for relevant documents.
        """
        try:
            collection = self.collection
            results = collection.query(query_texts=[prompt], n_results=n_results)
            return results
        except Exception as e:
            raise (f"Error happened while returning the relevant document from collection - {e}")
        
