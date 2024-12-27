"""
    RAG (Retrieval Augmented Generation) system for document Q&A.

    This module implements a RAG system using Streamlit, ChromaDB, and Ollama.
    It provides document processing, vector storage, and question answering capabilities.

"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

import ollama

from sentence_transformers import CrossEncoder


class DocumentProcessor:

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        """Handles document loading and text splitting operations.
        
        Attributes:
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between consecutive chunks
            separators (list): Text separators for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ".", "!", " ", ""]

    def process_document(self, uploaded_file: UploadedFile) -> list [Document]:
        """Process uploaded PDF file into chunks.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            list: List of Document objects containing text chunks
        """
        try:
            # store the uploaded file as temp file
            temp_file = tempfile.NamedTemporaryFile("wb", suffix='.pdf', delete=False)
            temp_file.write(uploaded_file.read())

            loader = PyMuPDFLoader(temp_file.name)
            docs = loader.load()
            
            os.unlink(temp_file.name) # delete the temp file

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap,
                separators = self.separators,
            )
            return text_splitter.split_documents(docs)
        except Exception as e:
            raise (f"Error happened while processing the uploaded document - {e}")
        

# ========================================================================================================================== #


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
            chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
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
        

# ========================================================================================================================== #


class LLMInterface:

    def __init__(self, model_name: str = 'llama3.2:3b'):
        """Handles interactions with the language model.
    
        Attributes:
            model_name (str): Name of the LLM model
            system_prompt (str): System prompt for the LLM
        """
        self.model_name = model_name
        self.system_prompt = """
                You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

                context will be passed as "Context:"
                user question will be passed as "Question:"

                To answer the question:
                1. Thoroughly analyze the context, identifying key information relevant to the question.
                2. Organize your thoughts and plan your response to ensure a logical flow of information.
                3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
                4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
                5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

                Format your response as follows:
                1. Use clear, concise language.
                2. Organize your answer into paragraphs for readability.
                3. Use bullet points or numbered lists where appropriate to break down complex information.
                4. If relevant, include any headings or subheadings to structure your response.
                5. Ensure proper grammar, punctuation, and spelling throughout your answer.

                Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
                """

    def re_rank_crossencoder(self, document: list[str], prompt) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

        Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
        their relevance to the query prompt. Returns the concatenated text of the top 3 most
        relevant documents along with their indices.

        Args:
            documents: List of document strings to be re-ranked.

        Returns:
            tuple: A tuple containing:
                - relevant_text (str): Concatenated text from the top 3 ranked documents
                - relevant_text_ids (list[int]): List of indices for the top ranked documents

        Raises:
            ValueError: If documents list is empty
            RuntimeError: If cross-encoder model fails to load or rank documents
        """
        try:
            relevant_text = ""
            relevant_text_ids = []

            encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            ranks = encoder_model.rank(prompt, document, top_k=3)
            for rank in ranks:
                relevant_text += document[rank['corpus_id']]
                relevant_text_ids.append(rank['corpus_id'])

            return relevant_text, relevant_text_ids
        except Exception as e:
            raise (f"Error happened while Re-ranks documents using a cross-encoder model - {e}")   
        
    def call_llm(self, context: str, prompt: str):
        """Calls a language model (LLM) to generate responses based on the provided context and prompt.

        This method interacts with an LLM using the Ollama API to obtain a response. It sends a system prompt along with user-defined context and question, and yields the generated response in chunks.

        Args:
            context: A string providing contextual information that informs the LLM's response.
            prompt: A string containing the user's question or query for which a response is sought.

        Yields:
            str: The generated content from the LLM in chunks. Each chunk is yielded as it is received.

        Raises:
            Exception: If there are issues during the interaction with the LLM, such as network errors or invalid parameters.

        Notes:
            - The method uses streaming to receive responses incrementally (stream = True), allowing for real-time updates.
        """
        try:
            response = ollama.chat(
                model = self.model_name,
                stream = True,
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}, Question: {prompt}",
                    }
                ],
            )

            for chunk in response:
                if chunk['done'] is False:
                    yield chunk['message']['content']
                else:
                    break
        except Exception as e:
            raise (f"Error happened while calling the LLM model to generate responses - {e}")    

# ========================================================================================================================== #


class RAGApplication:

    def __init__(self):
        """Main application class coordinating RAG components.
        
        Attributes:
            doc_processor (DocumentProcessor): Handles document processing
            vector_store (VectorStore): Manages vector storage
            llm_interface (LLMInterface): Handles LLM interactions
        """
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_interface = LLMInterface()

    def process_upload(self, uploaded_file: 'UploadedFile') -> None:
        """Processes an uploaded document and adds its contents to a vector collection.

        This method handles the processing of an uploaded file by first normalizing its name,
        extracting document splits using a document processor, and then adding those splits 
        to a vector collection for semantic search.

        Args:
            uploaded_file: An instance of `UploadedFile` representing the document that has been uploaded.

        Returns:
            None. The method performs actions to process the document and update the vector collection.

        Raises:
            Exception: If there are issues during document processing or adding to the vector collection.

        Notes:
            - The filename is sanitized by replacing hyphens, periods, and spaces with underscores.
        """
        try:
            file_name = uploaded_file.name.translate(str.maketrans({"-":"_", ".":"_", " ":"_"}))
            documents = self.doc_processor.process_document(uploaded_file)
            self.vector_store.add_to_vector_collection(documents, file_name)
        except Exception as e:
            raise (f"Error happened during document processing or adding to the vector collection - {e}")   
        
    def answer_question(self, prompt: str):
        """Generate an answer for the user's question based on vector store results.

        This method retrieves relevant documents from the vector store using the provided prompt,
        and then generates an answer by calling the language model interface with the most relevant
        document and the user's question.

        Args:
            prompt: A string containing the user's question or query for which an answer is sought.

        Returns:
            str: The generated answer from the language model, based on the most relevant document.

        Raises:
            IndexError: If no documents are found in the vector store for the given prompt.
            Exception: If there are issues during querying the vector store or calling the language model.

        Notes:
            - The method assumes that the vector store's query returns a dictionary containing a 'documents' key.
            - Ensure that both `vector_store` and `llm_interface` are properly initialized before calling this method.
        """
        try:
            results = self.vector_store.query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = self.llm_interface.re_rank_crossencoder(context, prompt)
            return results, relevant_text, relevant_text_ids, self.llm_interface.call_llm(context=relevant_text, prompt=prompt)
        except Exception as e:
            raise (f"Error happened during querying the vector store or calling the language model - {e}")   
        
def main():
    """Initialize and run the Streamlit application."""
    app = RAGApplication()
    st.title("RAG Question Answer System")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("**📑 Upload PDF files for Quries** ", type=['pdf'], accept_multiple_files=False)
        if uploaded_file and st.button("⚡️ Process"):
            app.process_upload(uploaded_file)
            st.success("Document processed!")

    prompt = st.text_area("**Ask a question related to your document:**")
    
    # prompt = st.text_input("Ask a question:")
    if prompt and st.button("Get Answer"):
        results, relevant_text, relevant_text_ids, response = app.answer_question(prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents :"):
            st.write(results)

        with st.expander("See most relevant document ids :"):
            st.write(relevant_text_ids)
            st.write(relevant_text)

if __name__ == "__main__":
    main()
