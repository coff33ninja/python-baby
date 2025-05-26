# extractors/__init__.py
import os
import mimetypes # Ensure mimetypes is imported
import logging # Use logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# Attempt to import extractors; they might not exist in the very first run of this subtask
# if they are created in the same subtask instruction set.
try:
    from .generic_extractor import GenericExtractor
except ImportError:
    logger.warning("GenericExtractor not found yet. Will be available after its creation.")
    GenericExtractor = None # Placeholder

try:
    from .jupyter_extractor import JupyterExtractor
except ImportError:
    logger.warning("JupyterExtractor not found yet. Will be available after its creation.")
    JupyterExtractor = None # Placeholder

try:
    from .pdf_extractor import PdfExtractor # For consistency, though it's next step
except ImportError:
    logger.warning("PdfExtractor not found yet. Will be available after its creation.")
    PdfExtractor = None # Placeholder


# Add common extensions if mimetypes doesn't know them well enough
mimetypes.add_type("application/vnd.jupyter", ".ipynb")
mimetypes.add_type("application/x-ipynb+json", ".ipynb") # Another common one
mimetypes.add_type("application/pdf", ".pdf")
# Add more as needed for other specific types

TEXT_EXTENSIONS = {
    ".py", ".js", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".swift", # Code
    ".html", ".htm", ".xml", ".json", ".yaml", ".yml", ".toml", # Markup/Data
    ".css", ".scss", ".less", # Stylesheets
    ".md", ".rst", ".txt", ".rtf", # Text/Markdown
    ".sh", ".bat", ".ps1", # Scripts
    ".sql", ".csv", ".tsv", # Data/SQL
    ".log", ".gitignore", ".gitattributes", ".env", ".ini", ".cfg", # Logs & Config
    ".dockerfile", "makefile", ".lua", ".rb", ".php", ".pl", ".scala", ".kts", ".groovy", ".dart",
    ".tex", ".bib", ".srt", ".vtt", # Various text formats
    # Add any other extensions known to be primarily text
}

# These are types that are binary but we might have specific extractors for.
# GenericExtractor should generally NOT try to decode these if is_likely_text=False
SPECIFICALLY_HANDLED_BINARY_EXTENSIONS = {".pdf", ".ipynb"} # Will expand with office docs, etc.
SPECIFICALLY_HANDLED_BINARY_MIMES = {"application/pdf", "application/vnd.jupyter", "application/x-ipynb+json"}


def extract_text_from_file(filepath: str) -> str | None:
    if not os.path.exists(filepath) or os.path.isdir(filepath):
        logger.warning(f"File does not exist or is a directory: {filepath}")
        return None

    mime_type, _ = mimetypes.guess_type(filepath)
    file_ext = os.path.splitext(filepath)[1].lower()
    
    text_content = None
    extractor_used = "None"

    try:
        if JupyterExtractor and file_ext == ".ipynb": # Check if class exists
            text_content = JupyterExtractor.extract_text(filepath)
            extractor_used = "JupyterExtractor"
        elif PdfExtractor and file_ext == ".pdf": # Check if class exists
            # This will be implemented in the next step, but good to have the path
            text_content = PdfExtractor.extract_text(filepath)
            extractor_used = "PdfExtractor"
        # Add elif for OfficeExtractor when implemented

        # Fallback for known text extensions or general text mimetypes
        elif file_ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
            if GenericExtractor: # Check if class exists
                text_content = GenericExtractor.extract_text(filepath, is_likely_text=True)
                extractor_used = "GenericExtractor (Text Type)"
            else:
                logger.warning("GenericExtractor is None, cannot process text type.")
        
        # For types not specifically text or handled by specific extractors above
        elif mime_type and not mime_type.startswith("text/") and \
             file_ext not in SPECIFICALLY_HANDLED_BINARY_EXTENSIONS and \
             mime_type not in SPECIFICALLY_HANDLED_BINARY_MIMES:
            logger.debug(f"Attempting generic extraction for non-text/unknown type: {filepath} (MIME: {mime_type})")
            if GenericExtractor: # Check if class exists
                text_content = GenericExtractor.extract_text(filepath, is_likely_text=False)
                extractor_used = "GenericExtractor (Attempted on Non-Text/Unknown)"
            else:
                logger.warning("GenericExtractor is None, cannot process non-text/unknown type.")
        else:
            logger.info(f"File type not explicitly supported or handled by generic for text extraction: {filepath} (MIME: {mime_type}, Ext: {file_ext})")

    except Exception as e:
        logger.error(f"Error extracting text from {filepath} using {extractor_used if text_content is None else 'detection'}: {e}", exc_info=True)
        return None

    if text_content and text_content.strip():
        logger.info(f"Successfully extracted text (length: {len(text_content)}) from {filepath} using {extractor_used}")
        return text_content.strip() # Return stripped non-empty content
    elif extractor_used != "None":
        logger.info(f"Extractor {extractor_used} returned no content for {filepath}")
        return None
    else: # No extractor was even attempted for this file type based on rules
        return None
