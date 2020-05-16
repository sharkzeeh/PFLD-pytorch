import requests
import os


WFLW_FOLDER = 'data/WFLW'
if not os.path.exists(WFLW_FOLDER):
    os.makedirs(WFLW_FOLDER)



def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_file_from_web_server(url, destination):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    response = requests.get(url, stream=True)
    save_response_content(response, os.path.join(destination, local_filename))

    return local_filename


#  TODO Add progress bar
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":


    train_images = 'WFLW_images.tar.gz'
    annots = 'WFLW_annotations.tar.gz'

    file_ids = ['1lFAxagOjvJQBiOsb6vNhtjAPd1tsB3CA',
                '1XoYjOxW-tnP73u81ANw4N3eDIMiLtxIy']   

    to_download = [train_images, annots]

    destinations = [WFLW_FOLDER + '/' + d for d in to_download]

    for file_id, destination in zip(file_ids, destinations):
        print(f"downloading {destination.split('/')[2]} from google drive...")
        download_file_from_google_drive(file_id, destination)
    print('all got downloaded!')