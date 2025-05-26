# extractors/pdf_extractor.py
import logging
try:
    import PyPDF2 # Will need to be added to requirements.txt
except ImportError:
    PyPDF2 = None # Allows module to be imported even if PyPDF2 is not installed yet

logger = logging.getLogger(__name__)

class PdfExtractor:
    @staticmethod
    def extract_text(filepath: str) -> str | None:
        """
        Extracts text from a PDF file using PyPDF2.
        Returns concatenated text from all pages, or None if extraction fails.
        """
        if PyPDF2 is None:
            logger.error("PyPDF2 library is not installed. PDF extraction is disabled.")
            return None

        try:
            extracted_texts = []
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                logger.debug(f"Extracting text from PDF {filepath} ({num_pages} pages).")
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    extracted_texts.append(page.extract_text() or "") # Ensure None from extract_text is handled
            
            full_text = "\n\n--- Page Break ---\n\n".join(extracted_texts)
            return full_text.strip() if full_text.strip() else None
        except FileNotFoundError:
            logger.error(f"File not found during PDF extraction: {filepath}", exc_info=True)
            return None
        except Exception as e:
            # PyPDF2 can raise various errors for corrupted or encrypted PDFs
            logger.error(f"Error extracting text from PDF {filepath}: {e}", exc_info=True)
            return None
