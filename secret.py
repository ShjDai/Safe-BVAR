import os
import random
from Crypto.Cipher import AES, Salsa20

import numpy as np
import torch

def generate_random_bits(bit_length=128) -> bytes:
    return os.urandom(bit_length // 8)

def bytes_to_binary_list(bytes_data: bytes) -> list[int]:
    bit_length = len(bytes_data) * 8
    ret = [int(bit) for bit in bin(int.from_bytes(bytes_data))[2:]]
    if len(ret) < bit_length:
        ret = [0] * (bit_length - len(ret)) + ret
    return ret

def binary_list_to_bytes(binary_list: list[int]) -> bytes:
    return int(''.join(str(bit) for bit in binary_list), 2).to_bytes(len(binary_list) // 8, 'big')

def encrypt_aes(key: bytes, plaintext: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(plaintext)

def decrypt_aes(key: bytes, ciphertext: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.decrypt(ciphertext)

def repeat_bytes(bytes_data: bytes, repeat_times: int) -> bytes:
    return bytes_data * repeat_times

def derepeat_bytes(bytes_data: bytes, repeat_times: int) -> bytes:
    return bytes_data[:len(bytes_data) // repeat_times]

def split_bytes(iterable_data, split_size: int) -> list:
    return [iterable_data[i:i+split_size] for i in range(0, len(iterable_data), split_size)]


MAX_TEXT_SIZE = 256
REPEAT_TIMES = 1

def text2latent(text: bytes, key: bytes, repeat_times=REPEAT_TIMES, max_text_size=MAX_TEXT_SIZE) -> list[int]:
    latent = []
    plaintext_splitted = split_bytes(text, max_text_size)
    print('plaintext_splitted size:', len(plaintext_splitted))
    for plaintext_chunk in plaintext_splitted:
        # print(len(plaintext_chunk))
        plaintext_repeated = repeat_bytes(plaintext_chunk, repeat_times)
        ciphertext = encrypt_aes(key, plaintext_repeated)
        cipher_latent = bytes_to_binary_list(ciphertext)
        # print('cipher_latent size:', len(cipher_latent))
        latent.extend(cipher_latent)
    return latent

def latent2text(latent: list[int], key: bytes, repeat_times=REPEAT_TIMES, max_text_size=MAX_TEXT_SIZE) -> bytes:
    decrypted_text_full = b''
    for latent_chunk in split_bytes(latent, max_text_size * repeat_times * 8):
        latent = binary_list_to_bytes(latent_chunk)
        decrypted_text_repeated = decrypt_aes(key, latent)
        decrypted_text = derepeat_bytes(decrypted_text_repeated, repeat_times)
        decrypted_text_full += decrypted_text
    return decrypted_text_full


def random_flip_bits(bits: list[int], flip_rate: float) -> list[int]:
    def flip_bit(b: int) -> int:
        if b == 0:
            return 1
        else:
            return 0
    return [flip_bit(b) if random.random() < flip_rate else b for b in bits]


def alice(secret: bytes, key: bytes) -> tuple[list[int], bytes]:
    secret_bits = torch.repeat_interleave(torch.tensor(bytes_to_binary_list(secret)), REPEAT_TIMES).tolist()
    cipher = Salsa20.new(key=key)
    msg_nonce = cipher.nonce
    ciphertext = cipher.encrypt(binary_list_to_bytes(secret_bits))
    return bytes_to_binary_list(ciphertext), msg_nonce

def bob(ciphertext: list[int], msg_nonce: bytes, key: bytes, repeat_times=REPEAT_TIMES) -> bytes:
    cipher = Salsa20.new(key=key, nonce=msg_nonce)
    ciphertext_flipped = binary_list_to_bytes(ciphertext)
    plaintext_flipped = bytes_to_binary_list(cipher.decrypt(ciphertext_flipped))
    plaintext_bits = []
    for chunk in split_bytes(plaintext_flipped, repeat_times):
        if sum(chunk) > repeat_times // 2:
            plaintext_bits.append(1)
        else:
            plaintext_bits.append(0)
    return binary_list_to_bytes(plaintext_bits)

def compare_bit_acc(secret: bytes, plaintext: bytes) -> float:
    secret_b = np.array(bytes_to_binary_list(secret))
    plaintext_b = np.array(bytes_to_binary_list(plaintext))
    return np.sum(secret_b == plaintext_b) / len(secret_b)


if __name__ == '__main__':
    key = generate_random_bits(256)
    
    with open('./test.txt', mode="a+") as f:
        f.write(key.hex())
        f.write('\n')

    # secret = binary_list_to_bytes([1]*16+[0]*16 + [1]*16+[0]*16 + [1]*32+[0]*32)
    secret = b'hello world. long long ago, there was a cat. salam. nihao. asdfg' * 8
    ciphertext_bits, msg_nonce = alice(secret, key)
    # secret_bits = torch.repeat_interleave(torch.tensor(bytes_to_binary_list(secret)), REPEAT_TIMES).tolist()
    # cipher = Salsa20.new(key=key)
    # msg_nonce = cipher.nonce
    # ciphertext = cipher.encrypt(binary_list_to_bytes(secret_bits))
    # print(len(secret_bits), sum(bytes_to_binary_list(msg)), len(msg))
    # msg_nonce = msg[:8]
    # ciphertext = msg[8:]
    ciphertext_flipped = random_flip_bits(ciphertext_bits, 0.2)
    plaintext_bits = bob(ciphertext_flipped, msg_nonce, key)
    print(len(ciphertext_flipped), len(plaintext_bits))
    # ciphertext_flipped = binary_list_to_bytes(ciphertext_flipped)
    # cipher = Salsa20.new(key=key, nonce=msg_nonce)
    # plaintext_flipped = bytes_to_binary_list(cipher.decrypt(ciphertext_flipped))
    # plaintext_bits = []
    # for chunk in split_bytes(plaintext_flipped, REPEAT_TIMES):
    #     if sum(chunk) > REPEAT_TIMES // 2:
    #         plaintext_bits.append(1)
    #     else:
    #         plaintext_bits.append(0)
    secret_b = np.array(bytes_to_binary_list(secret))
    plaintext_b = np.array(bytes_to_binary_list(plaintext_bits))
    print(np.sum(secret_b == plaintext_b) / len(secret_b))
    
    # print(plaintext == binary_list_to_bytes(secret_bits))
    
    # key = generate_random_bits(128)
    # plain_text = generate_random_bits(2**15)
    # latent = text2latent(plain_text, key)
    # print('latent size:', len(latent))
    # decrypted_text = latent2text(latent, key)
    # original_text = np.array(bytes_to_binary_list(plain_text))
    # decrypted_text = np.array(bytes_to_binary_list(decrypted_text))

    # print(np.sum(original_text == decrypted_text))
