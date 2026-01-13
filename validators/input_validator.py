from typing import Dict, Any, List, Optional
import re
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """
    Validator for user inputs and configuration values
    """

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Validate if a string is a properly formatted URL
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def is_valid_api_key(api_key: str, min_length: int = 10) -> bool:
        """
        Validate if an API key is in a valid format
        """
        if not api_key or len(api_key) < min_length:
            return False

        # Basic check: API keys are usually alphanumeric with possible special chars
        # This is a general check; specific services may have different requirements
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, api_key))

    @staticmethod
    def is_valid_collection_name(name: str) -> bool:
        """
        Validate Qdrant collection name
        """
        if not name or len(name) == 0:
            return False

        # Qdrant collection names should follow certain rules
        # Alphanumeric, underscores, hyphens, dots, but not starting with special chars
        pattern = r'^[a-zA-Z][a-zA-Z0-9._-]*$'
        return bool(re.match(pattern, name)) and len(name) <= 62

    @staticmethod
    def is_valid_chunk_size(size: int) -> bool:
        """
        Validate chunk size
        """
        return isinstance(size, int) and 100 <= size <= 10000  # Reasonable range

    @staticmethod
    def is_valid_chunk_overlap(overlap: int) -> bool:
        """
        Validate chunk overlap
        """
        return isinstance(overlap, int) and 0 <= overlap <= 1000  # Reasonable range

    @staticmethod
    def is_valid_crawl_delay(delay: float) -> bool:
        """
        Validate crawl delay
        """
        return isinstance(delay, (int, float)) and 0 <= delay <= 10  # Up to 10 seconds

    @staticmethod
    def is_valid_max_depth(depth: int) -> bool:
        """
        Validate max crawl depth
        """
        return isinstance(depth, int) and 1 <= depth <= 10  # Reasonable range

    @staticmethod
    def validate_docusaurus_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate the entire Docusaurus configuration
        """
        errors = []

        # Validate required fields
        required_fields = ['docusaurus_url']
        for field in required_fields:
            if field not in config or not config[field]:
                errors.append(f"Missing required field: {field}")

        # Validate URL if provided
        if 'docusaurus_url' in config and config['docusaurus_url']:
            if not InputValidator.is_valid_url(config['docusaurus_url']):
                errors.append(f"Invalid docusaurus_url: {config['docusaurus_url']}")

        # Validate API keys if provided
        api_key_fields = ['cohere_api_key', 'qdrant_api_key']
        for field in api_key_fields:
            if field in config and config[field]:
                if not InputValidator.is_valid_api_key(config[field]):
                    errors.append(f"Invalid {field} format")

        # Validate Qdrant URL if provided
        if 'qdrant_url' in config and config['qdrant_url']:
            if not InputValidator.is_valid_url(config['qdrant_url']):
                errors.append(f"Invalid qdrant_url: {config['qdrant_url']}")

        # Validate collection name if provided
        if 'qdrant_collection_name' in config and config['qdrant_collection_name']:
            if not InputValidator.is_valid_collection_name(config['qdrant_collection_name']):
                errors.append(f"Invalid qdrant_collection_name: {config['qdrant_collection_name']}")

        # Validate numeric values
        if 'chunk_size' in config and config['chunk_size'] is not None:
            if not InputValidator.is_valid_chunk_size(config['chunk_size']):
                errors.append(f"Invalid chunk_size: {config['chunk_size']}")

        if 'chunk_overlap' in config and config['chunk_overlap'] is not None:
            if not InputValidator.is_valid_chunk_overlap(config['chunk_overlap']):
                errors.append(f"Invalid chunk_overlap: {config['chunk_overlap']}")

        if 'crawl_delay' in config and config['crawl_delay'] is not None:
            if not InputValidator.is_valid_crawl_delay(config['crawl_delay']):
                errors.append(f"Invalid crawl_delay: {config['crawl_delay']}")

        if 'max_depth' in config and config['max_depth'] is not None:
            if not InputValidator.is_valid_max_depth(config['max_depth']):
                errors.append(f"Invalid max_depth: {config['max_depth']}")

        return errors

    @staticmethod
    def validate_document_chunk(chunk_data: Dict[str, Any]) -> List[str]:
        """
        Validate a document chunk
        """
        errors = []

        required_fields = ['content', 'source_url', 'document_hierarchy']
        for field in required_fields:
            if field not in chunk_data:
                errors.append(f"Missing required field in chunk: {field}")
            elif not chunk_data[field]:
                errors.append(f"Empty required field in chunk: {field}")

        # Validate content length
        if 'content' in chunk_data and chunk_data['content']:
            content = chunk_data['content']
            if len(content.strip()) == 0:
                errors.append("Content cannot be empty or whitespace only")
            elif len(content) > 10000:  # Reasonable limit
                errors.append(f"Content too long: {len(content)} characters (max: 10000)")

        # Validate URL if provided
        if 'source_url' in chunk_data and chunk_data['source_url']:
            if not InputValidator.is_valid_url(chunk_data['source_url']):
                errors.append(f"Invalid source_url in chunk: {chunk_data['source_url']}")

        # Validate embedding if provided
        if 'embedding' in chunk_data and chunk_data['embedding']:
            embedding = chunk_data['embedding']
            if not isinstance(embedding, list):
                errors.append("Embedding must be a list of numbers")
            else:
                if not all(isinstance(x, (int, float)) for x in embedding):
                    errors.append("Embedding must contain only numeric values")

        return errors

    @staticmethod
    def validate_crawl_parameters(params: Dict[str, Any]) -> List[str]:
        """
        Validate crawl parameters
        """
        errors = []

        # Validate URL
        if 'url' in params and params['url']:
            if not InputValidator.is_valid_url(params['url']):
                errors.append(f"Invalid URL: {params['url']}")

        # Validate depth
        if 'max_depth' in params and params['max_depth'] is not None:
            if not InputValidator.is_valid_max_depth(params['max_depth']):
                errors.append(f"Invalid max_depth: {params['max_depth']}")

        # Validate delay
        if 'delay' in params and params['delay'] is not None:
            if not InputValidator.is_valid_crawl_delay(params['delay']):
                errors.append(f"Invalid delay: {params['delay']}")

        return errors

    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """
        Sanitize input string to prevent injection attacks
        """
        if not input_str:
            return ""

        # Remove null bytes and other potentially dangerous characters
        sanitized = input_str.replace('\x00', '').strip()

        # Additional sanitization can be added based on specific requirements
        return sanitized

    @staticmethod
    def validate_environment_config(env_vars: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Validate environment configuration variables
        """
        validation_results = {}

        # Check for required environment variables
        required_vars = ['DOCUSAURUS_URL', 'COHERE_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
        missing_vars = []
        for var in required_vars:
            if not env_vars.get(var):
                missing_vars.append(var)

        if missing_vars:
            validation_results['missing_required'] = missing_vars

        # Validate specific variables
        validation_errors = []

        # Validate URLs
        if env_vars.get('DOCUSAURUS_URL') and not InputValidator.is_valid_url(env_vars['DOCUSAURUS_URL']):
            validation_errors.append(f"Invalid DOCUSAURUS_URL: {env_vars['DOCUSAURUS_URL']}")

        if env_vars.get('QDRANT_URL') and not InputValidator.is_valid_url(env_vars['QDRANT_URL']):
            validation_errors.append(f"Invalid QDRANT_URL: {env_vars['QDRANT_URL']}")

        # Validate API keys
        if env_vars.get('COHERE_API_KEY') and not InputValidator.is_valid_api_key(env_vars['COHERE_API_KEY']):
            validation_errors.append("Invalid COHERE_API_KEY format")

        if env_vars.get('QDRANT_API_KEY') and not InputValidator.is_valid_api_key(env_vars['QDRANT_API_KEY']):
            validation_errors.append("Invalid QDRANT_API_KEY format")

        # Validate collection name
        collection_name = env_vars.get('QDRANT_COLLECTION_NAME', 'docusaurus_chunks')
        if not InputValidator.is_valid_collection_name(collection_name):
            validation_errors.append(f"Invalid QDRANT_COLLECTION_NAME: {collection_name}")

        # Validate numeric values if provided
        chunk_size = env_vars.get('CHUNK_SIZE')
        if chunk_size and not InputValidator.is_valid_chunk_size(int(chunk_size) if chunk_size.isdigit() else -1):
            validation_errors.append(f"Invalid CHUNK_SIZE: {chunk_size}")

        chunk_overlap = env_vars.get('CHUNK_OVERLAP')
        if chunk_overlap and not InputValidator.is_valid_chunk_overlap(int(chunk_overlap) if chunk_overlap.isdigit() else -1):
            validation_errors.append(f"Invalid CHUNK_OVERLAP: {chunk_overlap}")

        crawl_delay = env_vars.get('CRAWL_DELAY')
        if crawl_delay:
            try:
                delay_val = float(crawl_delay)
                if not InputValidator.is_valid_crawl_delay(delay_val):
                    validation_errors.append(f"Invalid CRAWL_DELAY: {crawl_delay}")
            except ValueError:
                validation_errors.append(f"Invalid CRAWL_DELAY value: {crawl_delay}")

        max_depth = env_vars.get('MAX_DEPTH')
        if max_depth and not InputValidator.is_valid_max_depth(int(max_depth) if max_depth.isdigit() else -1):
            validation_errors.append(f"Invalid MAX_DEPTH: {max_depth}")

        if validation_errors:
            validation_results['validation_errors'] = validation_errors

        return validation_results