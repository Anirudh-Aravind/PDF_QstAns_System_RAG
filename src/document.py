import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

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
        