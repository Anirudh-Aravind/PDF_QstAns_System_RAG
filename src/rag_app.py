from src.document import DocumentProcessor
from src.vector_store import VectorStore
from src.llm import LLMInterface

from streamlit.runtime.uploaded_file_manager import UploadedFile

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