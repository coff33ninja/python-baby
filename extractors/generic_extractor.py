# extractors/generic_extractor.py
import chardet # For guessing encoding
import json # For parsing potential JSON content
import logging
import os # Added for os.path.getsize

logger = logging.getLogger(__name__)

class GenericExtractor:
    COMMON_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

    @staticmethod
    def _extract_strings_from_json_data(data) -> list[str]:
        """
        Recursively extracts all string values from a parsed JSON structure.
        """
        strings = []
        if isinstance(data, dict):
            for _k, v in data.items():
                # Optionally, could include keys if they are also considered text:
                # if isinstance(k, str):
                #     strings.append(k)
                if isinstance(v, str):
                    strings.append(v)
                else:
                    strings.extend(GenericExtractor._extract_strings_from_json_data(v))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    strings.append(item)
                else:
                    strings.extend(GenericExtractor._extract_strings_from_json_data(item))
        return strings
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

                    # Attempt to parse as JSON if it's likely text or a small "maybe text" file
                    # Max size for JSON attempt on "maybe text" files, e.g., 1MB
                    attempt_json_parse = is_likely_text or (file_size < 1 * 1024 * 1024)

                    if content and attempt_json_parse:
                        try:
                            parsed_json = json.loads(content)
                            logger.debug(f"Successfully parsed content of {filepath} as JSON (decoded with {encoding}).")
                            extracted_strings = GenericExtractor._extract_strings_from_json_data(parsed_json)
                            if extracted_strings:
                                json_derived_text = " ".join(extracted_strings).strip()
                                if json_derived_text: # Ensure there's actual text
                                    logger.info(f"Extracted text from JSON structure in {filepath} (decoded with {encoding}).")
                                    return json_derived_text
                                else:
                                    logger.debug(f"JSON structure in {filepath} yielded no strings. Falling back to decoded text.")
                            else:
                                logger.debug(f"JSON structure in {filepath} (decoded with {encoding}) yielded no strings. Falling back to decoded text.")
                        except json.JSONDecodeError:
                            logger.debug(f"Content of {filepath} (decoded with {encoding}) is not valid JSON. Treating as plain text.")
                        # If JSON parsing fails or yields no text, fall through to treat `content` as plain text.

                    # Fallback or standard text processing if not handled as JSON / JSON attempt failed
                    if content: # Check if content is not empty after decode
                        # Heuristic for non-likely text files: check for excessive null bytes
                        if not is_likely_text:
                            null_bytes_ratio = content.count('\x00') / len(content) if len(content) > 0 else 0
                            if null_bytes_ratio > 0.1: # If more than 10% null bytes, likely not text
                                logger.debug(f"File {filepath} decoded with {encoding} but has high null byte ratio ({null_bytes_ratio:.2f}). Likely not text.")
                                continue # Try next encoding

                        logger.debug(f"Successfully decoded {filepath} with {encoding} (treated as plain text).")
                        stripped_content = content.strip()
                        return stripped_content if stripped_content else None # Return None if stripping results in empty

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
