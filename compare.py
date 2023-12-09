import os
import requests
from PIL import Image
from io import BytesIO
from search_wardrobe import search_wardrobe


def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Error downloading image: Status code {response.status_code}")


def compare_image_with_wardrobe(image_url, threshold=0.5):
    try:
        image = download_image(image_url)
        image_path = './temp_image.jpg'
        image.save(image_path)

        results = search_wardrobe(image_path, threshold=threshold)
        os.remove(image_path)

        return results
    except Exception as e:
        return f"An error occurred: {str(e)}"


image_url = 'https://www.realmenrealstyle.com/wp-content/uploads/2023/09/2nd-ranking-factor.jpg'
print(compare_image_with_wardrobe(image_url))
