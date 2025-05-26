# extractors/jupyter_extractor.py
import json
import nbformat # Will need to be added to requirements.txt
import logging

logger = logging.getLogger(__name__)

class JupyterExtractor:
    @staticmethod
    def extract_text(filepath: str) -> str | None:
        """
        Extracts text from code and markdown cells in a Jupyter notebook.
        Returns concatenated text as a string, or None if extraction fails.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # nbformat.read is preferred over json.load for notebooks
                # as it validates notebook structure.
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT) 
            
            extracted_texts = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    extracted_texts.append(cell.get('source', ''))
                elif cell.get('cell_type') == 'code':
                    # Could also include outputs if desired, but source is primary for now
                    extracted_texts.append(cell.get('source', ''))
            
            full_text = "\n\n---\n\n".join(extracted_texts) # Separate cells clearly
            return full_text.strip() if full_text.strip() else None
        except FileNotFoundError:
            logger.error(f"File not found during Jupyter notebook extraction: {filepath}", exc_info=True)
            return None
        except nbformat.reader.NotJSONError:
             logger.error(f"File {filepath} is not a valid JSON file, cannot parse as notebook.", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Error extracting text from Jupyter notebook {filepath}: {e}", exc_info=True)
            return None
