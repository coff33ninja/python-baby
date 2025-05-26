# extractors/jupyter_extractor.py
import nbformat # Will need to be added to requirements.txt
import logging
import os # Added for os.path.getsize

logger = logging.getLogger(__name__)

class JupyterExtractor:

    @staticmethod
    def _get_text_from_cell_field(field_data) -> str:
        """
        Safely extracts and joins text from a notebook cell field,
        which can be a string or a list of strings.
        """
        if isinstance(field_data, str):
            return field_data
        elif isinstance(field_data, list):
            return "".join(str(s) for s in field_data) # Ensure all elements are strings before joining
        return ""

    @staticmethod
    def _extract_strings_from_json_object(data_obj) -> list[str]:
        """
        Recursively extracts all string values from a Python object
        that was originally parsed from JSON (e.g., dicts, lists, strings).
        This is similar to GenericExtractor._extract_strings_from_json_data.
        """
        strings = []
        if isinstance(data_obj, dict):
            for _k, v in data_obj.items():
                # Optionally extract keys if they are also considered text:
                # if isinstance(_k, str): strings.append(_k)
                if isinstance(v, str):
                    strings.append(v)
                else:
                    strings.extend(JupyterExtractor._extract_strings_from_json_object(v))
        elif isinstance(data_obj, list):
            for item in data_obj:
                if isinstance(item, str):
                    strings.append(item)
                else:
                    strings.extend(JupyterExtractor._extract_strings_from_json_object(item))
        elif isinstance(data_obj, str): # If the top-level data_obj itself is a string (e.g. simple JSON string value)
            strings.append(data_obj)
        # Ignores numbers, booleans, None if they are not part of a string.
        return strings

    @staticmethod
    def extract_text(filepath: str, max_size_mb: int = 10) -> str | None:
        """
        Extracts text from markdown, code, and raw cells, as well as code cell outputs
        in a Jupyter notebook.
        Returns concatenated text as a string, or None if extraction fails.
        """
        try:
            file_size = os.path.getsize(filepath)
            if file_size > max_size_mb * 1024 * 1024:
                logger.warning(f"File {filepath} (size {file_size}B) exceeds max size {max_size_mb}MB for Jupyter extraction. Skipping.")
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                # nbformat.read is preferred over json.load for notebooks
                # as it validates notebook structure.
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            extracted_texts = []
            cell_separator = "\n\n--- Cell End ---\n\n"
            output_separator = "\n\n--- Output --- \n"
            # Add enumerate back to get cell_num
            for cell_num, cell in enumerate(notebook.get('cells', [])):
                cell_type = cell.get('cell_type')
                source_text = JupyterExtractor._get_text_from_cell_field(cell.get('source', ''))

                if cell_type in ['markdown', 'raw']:
                    if source_text:
                        extracted_texts.append(f"### Cell {cell_num+1} ({cell_type}) ###\n{source_text}")
                elif cell_type == 'code':
                    if source_text:
                        extracted_texts.append(f"### Cell {cell_num+1} (code source) ###\n{source_text}")

                    outputs_text = []
                    for output in cell.get('outputs', []):
                        output_type = output.get('output_type')
                        if output_type == 'stream': # stdout, stderr
                            outputs_text.append(JupyterExtractor._get_text_from_cell_field(output.get('text', '')))
                        elif output_type in ['display_data', 'execute_result']:
                            data = output.get('data', {})
                            # Prioritize text/plain
                            if 'text/plain' in data:
                                outputs_text.append(JupyterExtractor._get_text_from_cell_field(data['text/plain']))
                            # Fallback to extracting strings from application/json if text/plain is not present or empty
                            elif 'application/json' in data:
                                json_content = data['application/json']
                                extracted_json_strings = JupyterExtractor._extract_strings_from_json_object(json_content)
                                outputs_text.append(" ".join(extracted_json_strings).strip())
                        elif output_type == 'error':
                            error_name = output.get('ename', 'Error')
                            error_value = output.get('evalue', '')
                            traceback_text = "\n".join(output.get('traceback', []))
                            outputs_text.append(f"Error: {error_name}: {error_value}\n{traceback_text}")

                    if outputs_text:
                        # Join all outputs from this code cell first, then append as a block
                        full_output_block = f"{output_separator}".join(filter(None, outputs_text))
                        if full_output_block.strip(): # Ensure there's actual text in the outputs
                             extracted_texts.append(f"### Cell {cell_num+1} (code outputs) ###{output_separator}{full_output_block.strip()}")

            if not extracted_texts:
                logger.info(f"No text content extracted from Jupyter notebook {filepath}.")
                return None

            full_text = cell_separator.join(filter(None, extracted_texts)) # Join all collected text blocks
            stripped_full_text = full_text.strip()
            logger.info(f"Successfully extracted text (length: {len(stripped_full_text)}) from Jupyter notebook {filepath}")
            return stripped_full_text if stripped_full_text else None

        except FileNotFoundError:
            logger.error(f"File not found during Jupyter notebook extraction: {filepath}", exc_info=True)
            return None
        except nbformat.reader.NotJSONError:
             logger.error(f"File {filepath} is not a valid JSON file, cannot parse as notebook.", exc_info=True)
             return None
        except UnicodeDecodeError as ude:
            logger.error(f"Unicode decode error for Jupyter notebook {filepath}. Ensure it's UTF-8. Error: {ude}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error extracting text from Jupyter notebook {filepath}: {e}", exc_info=True)
            return None
