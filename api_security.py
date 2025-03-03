# Create a new file: api_security.py

import base64
import hashlib
import json
import logging
import os
from getpass import getpass
from pathlib import Path

logger = logging.getLogger("ApiSecurity")


def encrypt_simple(text, password):
    """Simple encryption for API keys - not production grade but better than plaintext"""
    if not text:
        return ""

    password_hash = hashlib.sha256(password.encode()).digest()
    result = bytearray()

    for i, char in enumerate(text.encode()):
        key_char = password_hash[i % len(password_hash)]
        result.append((char + key_char) % 256)

    return base64.b64encode(result).decode()


def decrypt_simple(encrypted_text, password):
    """Simple decryption for API keys"""
    if not encrypted_text:
        return ""

    try:
        encrypted_bytes = base64.b64decode(encrypted_text.encode())
        password_hash = hashlib.sha256(password.encode()).digest()
        result = bytearray()

        for i, char in enumerate(encrypted_bytes):
            key_char = password_hash[i % len(password_hash)]
            result.append((char - key_char) % 256)

        return result.decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return ""


def load_api_keys(config_file="api_config.json", ask_password=True):
    """Load API keys safely from environment variables or encrypted config"""
    # First check environment variables (preferred method)
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if api_key and api_secret:
        logger.info("Using API keys from environment variables")
        return api_key, api_secret

    # If not in environment, try the config file
    config_path = Path(config_file)
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            encrypted_key = config.get('encrypted_api_key', '')
            encrypted_secret = config.get('encrypted_api_secret', '')

            if encrypted_key and encrypted_secret:
                if ask_password:
                    password = getpass("Enter password to decrypt API keys: ")
                else:
                    # Default password for non-interactive environments
                    # Not secure for production!
                    password = "trading_system"

                api_key = decrypt_simple(encrypted_key, password)
                api_secret = decrypt_simple(encrypted_secret, password)

                if api_key and api_secret:
                    logger.info("API keys loaded from encrypted config")
                    return api_key, api_secret
        except Exception as e:
            logger.error(f"Error loading API keys from config: {e}")

    logger.warning("No API keys found in environment or config")
    return "", ""


def save_api_keys(api_key, api_secret, config_file="api_config.json", password=None):
    """Save API keys to encrypted config file"""
    if not password:
        password = getpass("Enter password to encrypt API keys: ")
        password_confirm = getpass("Confirm password: ")
        if password != password_confirm:
            logger.error("Passwords do not match")
            return False

    encrypted_key = encrypt_simple(api_key, password)
    encrypted_secret = encrypt_simple(api_secret, password)

    config = {
        'encrypted_api_key': encrypted_key,
        'encrypted_api_secret': encrypted_secret
    }

    try:
        with open(config_file, 'w') as f:
            json.dump(config, f)
        logger.info(f"API keys encrypted and saved to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving encrypted API keys: {e}")
        return False


def get_api_credentials():
    """Single entry point for getting API credentials from all possible sources"""
    # First check environment variables (highest priority)
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if api_key and api_secret:
        logger.info("Using API keys from environment variables")
        return api_key, api_secret

    # Then try config file with encryption
    api_key, api_secret = load_api_keys(ask_password=False)
    if api_key and api_secret:
        return api_key, api_secret

    # Finally return empty values
    logger.warning("No API credentials found")
    return "", ""