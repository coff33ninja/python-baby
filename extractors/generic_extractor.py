# extractors/generic_extractor.py
import chardet # For guessing encoding
import logging
import os # Added for os.path.getsize

logger = logging.getLogger(__name__)

class GenericExtractor:
    COMMON_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

    @staticmethod
    def extract_text(filepath: str, is_likely_text: bool = True, max_size_mb: int = 10) -> str | None:
        """
        Extracts text from presumably text-based files.
        Tries common encodings.
        If is_likely_text is False, it's more cautious and might only try UTF-8
        or skip if file size is very large.
        Returns None if extraction fails or content is empty after strip.
        """
        try:
            file_size = os.path.getsize(filepath)
            if file_size > max_size_mb * 1024 * 1024 :
                logger.warning(f"File {filepath} (size {file_size}B) exceeds max size {max_size_mb}MB for generic extraction. Skipping.")
                return None

            # For files not explicitly marked as text, be more careful
            if not is_likely_text and file_size > 1 * 1024 * 1024: # Stricter limit for "maybe text" files
                logger.warning(f"File {filepath} (size {file_size}B) is large and not marked as likely text. Skipping generic attempt.")
                return None

            with open(filepath, 'rb') as f_binary:
                raw_bytes = f_binary.read()

            detected_encoding = None
            if is_likely_text or file_size < 200 * 1024 : # Only run chardet on smaller "maybe text" files
                try:
                    detected_encoding_result = chardet.detect(raw_bytes[:10000]) # Detect on a sample
                    if detected_encoding_result and detected_encoding_result['confidence'] > 0.5:
                        detected_encoding = detected_encoding_result['encoding']
                        logger.debug(f"Chardet detected encoding: {detected_encoding} with confidence {detected_encoding_result['confidence']:.2f} for {filepath}")
                except Exception as chardet_err:
                    logger.debug(f"Chardet detection failed for {filepath}: {chardet_err}")


            encodings_to_try = list(GenericExtractor.COMMON_ENCODINGS)
            if detected_encoding and detected_encoding.lower() not in [enc.lower() for enc in encodings_to_try]:
                encodings_to_try.insert(0, detected_encoding) # Prioritize detected encoding

            # For files not flagged as likely text, be more conservative with encodings
            if not is_likely_text:
                encodings_to_try = [detected_encoding] if detected_encoding else ['utf-8']


            for encoding in encodings_to_try:
                try:
                    content = raw_bytes.decode(encoding)
                    # Further check: sometimes decode works but results in garbage if it's truly binary
                    # A simple heuristic: if it's not likely text, check for excessive null bytes or non-printable chars
                    if not is_likely_text and content:
                        null_bytes_ratio = content.count('\x00') / len(content) if len(content) > 0 else 0
                        if null_bytes_ratio > 0.1: # If more than 10% null bytes, likely not text
                            logger.debug(f"File {filepath} decoded with {encoding} but has high null byte ratio ({null_bytes_ratio:.2f}). Likely not text.")
                            continue # Try next encoding or fail

                    logger.debug(f"Successfully decoded {filepath} with {encoding}")
                    return content.strip() if content.strip() else None
                except UnicodeDecodeError:
                    logger.debug(f"Failed to decode {filepath} with {encoding}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error decoding {filepath} with {encoding}: {e}", exc_info=True)
                    return None # Stop on other errors

            logger.warning(f"Could not decode {filepath} with any attempted encodings: {encodings_to_try}")
            return None

        except FileNotFoundError:
            logger.error(f"File not found during generic extraction: {filepath}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error in GenericExtractor for {filepath}: {e}", exc_info=True)
            return None
