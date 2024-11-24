import os
import logging
from glob import glob
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CreateChunks:
    def __init__(self, pdf_directory: str) -> None:
        """
        Initializes the CreateChunks object with the specified PDF directory.

        Args:
            pdf_directory (str): The directory containing PDF files to process.
        """
        self.pdf_directory = glob(os.path.join(pdf_directory, '*.pdf'))  # Compatible with only PDF documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("chunk_size")),
            chunk_overlap=int(os.getenv("chunk_overlap")),
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

    def parse_pdf(self) -> list:
        """
        Parses all PDF files in the specified directory and extracts text.

        Returns:
            list: A list of dictionaries containing policy names and extracted text from each PDF.
        """
        parsed_text_dict = []
        for pdf_doc in self.pdf_directory:
            try:
                logging.info(f"Processing PDF: {pdf_doc}")
                with fitz.open(pdf_doc) as doc:
                    extracted_text = ""
                    for page in doc:
                        extracted_text += page.get_text()
                parsed_text_dict.append({
                    "policy_name": self._clean_policy_name(pdf_doc),
                    "pdf_name": pdf_doc.split('/')[-1],
                    "extracted_text": extracted_text
                })
            except Exception as e:
                logging.error(f"Error processing {pdf_doc}: {e}")
        return parsed_text_dict
    
    def _clean_policy_name(self, pdf_doc: str) -> str:
        """
        Cleans and formats the policy name extracted from the PDF file name.

        Args:
            pdf_doc (str): The path to the PDF file.

        Returns:
            str: A cleaned and formatted policy name.
        """
        return pdf_doc.split('/')[-1].split('.')[0].replace('-', '').replace('GPT', '').strip().lower()

    def create_chunks(self, extracted_texts: list) -> list:
        """
        Creates text chunks from the extracted texts using a text splitter.

        Args:
            extracted_texts (list): A list of dictionaries containing extracted text and policy names.

        Returns:
            list: A list of chunked documents created from the extracted texts.
        """
        chunked_docs = []
        for policy_doc in extracted_texts:
            try:
                doc = self.text_splitter.create_documents(
                    [policy_doc["extracted_text"]],
                    metadatas=[{"policy_name": policy_doc["policy_name"], "pdf_name": policy_doc["pdf_name"]}]
                )
                chunked_docs.extend(doc)
            except Exception as e:
                logging.error(f"Error creating chunks for {policy_doc['policy_name']}: {e}")
        return chunked_docs

    def get_chunks(self) -> list:
        """
        Main method to extract text from PDFs and create chunks.

        Returns:
            list: A list of chunked documents created from all processed PDFs.
        """
        extracted_texts = self.parse_pdf()
        return self.create_chunks(extracted_texts)

if __name__ == "__main__":
    pdf_directory = os.getenv("PDF_DIRECTORY", "../../../DS_Assignment/data/")
    obj = CreateChunks(pdf_directory=pdf_directory)
    
    try:
        chunks = obj.get_chunks()
        logging.info(f"Successfully created {len(chunks)} chunks.")
    except Exception as e:
        logging.error(f"An error occurred while getting chunks: {e}")