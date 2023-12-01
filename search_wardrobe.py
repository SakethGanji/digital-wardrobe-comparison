import numpy as np
import faiss
import sys

from index_builder import *
from nlp.mini_chat_3b import *

def one_hot_encode(feature, categories):
    vector = np.zeros(len(categories), dtype='float32')
    if feature in categories:
        vector[categories.index(feature)] = 1.0
    return vector

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


def is_category_compatible(item1, item2):
    return item1['subCategory'] != item2['subCategory']

index = faiss.read_index("wardrobe_index.faiss")
image_ids = []
with open("image_ids.txt", "r") as f:
    for line in f:
        id, path = line.strip().split(',')
        image_ids.append((int(id), path))

def search(query, k=10):
    query_features = interpret_query(query)
    query_vector = vectorize_features(query_features)
    D, I = index.search(np.array([query_vector]), k)
    return [image_ids[i] for i in I[0]]

def interpret_query(query):
    response = generate_response(query)

    features = {
        'gender': 'Unisex',
        'masterCategory': 'Apparel',
        'subCategory': 'Topwear',
        'articleType': 'Tshirts',
        'baseColour': 'Black',
        'season': 'Summer',
        'usage': 'Casual',
    }

    return features

def combine_items_into_outfits(search_results):
    outfits = []
    processed_combinations = set()

    for i in range(len(search_results)):
        for j in range(i + 1, len(search_results)):
            combination = (search_results[i], search_results[j])

            if combination in processed_combinations:
                continue

            item1 = extract_features(search_results[i][1])
            item2 = extract_features(search_results[j][1])

            if is_category_compatible(item1, item2):
                outfits.append(combination)
                processed_combinations.add(combination)

    return outfits


query = "I need an outfit for my presentation"
search_results = search(query)
combined_outfits = combine_items_into_outfits(search_results)
print("Combined Outfits:", combined_outfits)
