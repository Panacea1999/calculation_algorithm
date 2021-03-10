#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: one-time pad.py
@Abstract: one-time pad encryption
@Time: 2021/03/11 07:38:04
@Requirements: None
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''

from secrets import token_bytes
from typing import Tuple

def random_key(len):
    # generate a random key in form of bytes
    tmp_key = token_bytes(len)
    return int.from_bytes(tmp_key, 'big')

def encrypt(origin):
    # encode the data
    origin_bytes = origin.encode()
    # generate the key
    dummy_key = random_key(len(origin))
    origin_key = int.from_bytes(origin_bytes, 'big')
    encrypted = origin_key ^ dummy_key #XOR
    return dummy_key, encrypted

def decrypt(dummy_key, encrypted):
    decrypted = dummy_key ^ encrypted
    tmp_data = decrypted.to_bytes((decrypted.bit_length() + 7) // 8, 'big')
    return tmp_data.decode()

# test
if __name__ == '__main__':
    key1, key2 = encrypt('WangZy')
    res = decrypt(key1, key2)
    print('Dummy key: ', key1)
    print('Encrypted data: ', key2)
    print('Decrypted data: ', res)