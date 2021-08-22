import requests
import shutil
import os
from tqdm import tqdm


CHUNK_SIZE = 32768
URL = "https://docs.google.com/uc?export=download"


def download_file_from_google_drive(id, destination, total_size):
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination, total_size)
    unpack_remove_archive(destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def unpack_remove_archive(file_name):
    print(f"Now unpacking file '{file_name}'")
    shutil.unpack_archive(file_name, './')
    os.remove(file_name)

def save_response_content(response, destination, total_size):
    print(f"Now downloading file '{destination}'")
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total_size // CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)