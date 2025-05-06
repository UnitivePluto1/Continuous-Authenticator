from Crypto.Cipher import AES
import base64
import os

SECRET_KEY = os.environ.get("AES_SECRET_KEY", "Whyareyougayhuh?").encode()  # Ensure bytes

def pad(data):
    """Pad data to be a multiple of 16 bytes."""
    return data + (16 - len(data) % 16) * chr(16 - len(data) % 16)

def unpad(data):
    """Remove padding."""
    return data[:-ord(data[-1])]

def encrypt(data):
    """Encrypt data using AES."""
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    encrypted_bytes = cipher.encrypt(pad(data).encode())  # Ensure encoding
    return base64.b64encode(encrypted_bytes).decode()  # Convert to string

def decrypt(enc_data):
    """Decrypt AES-encrypted data."""
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    decrypted_bytes = cipher.decrypt(base64.b64decode(enc_data))  # Decode base64
    return unpad(decrypted_bytes.decode())  # Decode string
