import os
import numpy as np
import faiss

from data import CustomDataset, get_num_classes
from predict import load_model, load_label_mappings, predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(BASE_DIR, 'styles.csv')
model_path = os.path.join(BASE_DIR, 'saved_models', 'efficientnet_multioutput_model.pth')
label_mappings_path = os.path.join(BASE_DIR, 'label_mappings.json')

dataset = CustomDataset(csv_file=csv_file_path, data_dir='/workspace/digital-wardrobe-recommendation')
num_classes_dict = get_num_classes(dataset.data_frame)
model = load_model(model_path, num_classes_dict)
label_mappings = load_label_mappings(label_mappings_path)


def extract_features(image_path):
    if not image_path:
        return {"error": "Please provide an 'image_path' field in the request body."}

    if not os.path.isfile(image_path):
        return {"error": "File does not exist. Please enter a valid path."}

    predictions = predict(image_path, model, num_classes_dict, label_mappings)

    return predictions


def one_hot_encode(feature, categories):
    vector = np.zeros(len(categories), dtype='float32')
    if feature in categories:
        vector[categories.index(feature)] = 1.0
    return vector


gender_categories = ['Men', 'Women', 'Unisex', 'Boys', 'Girls']
master_category_categories = ['Apparel', 'Footwear', 'Accessories', 'Home', 'Personal Care', 'Sporting Goods']
sub_category_categories = ['Topwear', 'Bottomwear', 'Footwear', 'Accessories', 'Bags', 'Belts', 'Cufflinks', 'Eyewear',
                           'Headwear', 'Scarves', 'Socks', 'Ties', 'Wallets', 'Watches']
article_type_categories = ['Tshirts', 'Shirts', 'Jeans', 'Dresses', 'Trousers', 'Skirts', 'Sweaters', 'Jackets',
                           'Blazers', 'Shorts', 'Tracksuits']
base_colour_categories = ['Blue', 'Black', 'White', 'Red', 'Green', 'Yellow', 'Purple', 'Pink', 'Orange', 'Brown',
                          'Grey', 'Beige', 'Maroon']
season_categories = ['Summer', 'Fall', 'Winter', 'Spring']
usage_categories = ['Casual', 'Sports', 'Formal', 'Ethnic', 'Party', 'Home', 'Travel', 'Smart Casual']


def vectorize_features(features):
    vectors = []
    vectors.extend(one_hot_encode(features['gender'], gender_categories))
    vectors.extend(one_hot_encode(features['masterCategory'], master_category_categories))
    vectors.extend(one_hot_encode(features['subCategory'], sub_category_categories))
    vectors.extend(one_hot_encode(features['articleType'], article_type_categories))
    vectors.extend(one_hot_encode(features['baseColour'], base_colour_categories))
    vectors.extend(one_hot_encode(features['season'], season_categories))
    vectors.extend(one_hot_encode(features['usage'], usage_categories))
    return np.array(vectors, dtype='float32')


dimension = len(gender_categories) + len(master_category_categories) + len(sub_category_categories) + len(
    article_type_categories) + len(base_colour_categories) + len(season_categories) + len(usage_categories)

index = faiss.IndexFlatL2(dimension)

image_directory = "./wardrobe"
image_ids = []
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if
               filename.endswith(('.png', '.jpg', '.jpeg'))]

for i, image_path in enumerate(image_paths):
    features = extract_features(image_path)
    vector = vectorize_features(features)
    index.add(np.array([vector]))
    image_ids.append((i, image_path))

faiss.write_index(index, "wardrobe_index.faiss")
with open("image_ids.txt", "w") as f:
    for id, path in image_ids:
        f.write(f"{id},{path}\n")
