# script to unite the class for further evaluation
import json
import csv
import nltk
import spacy
from nltk.corpus import wordnet as wn

# --- Setup ---
nlp = spacy.load("en_core_web_md")  # needs: python -m spacy download en_core_web_md

# --- Config paths ---
imagenet_json = r"E:\Thesis\IRTNet\data\ImageNet\imagenet_class_index.json"
output_csv    = r"E:\Thesis\IRTNet\data\ImageNet\imagenet_to_cifar_superclasses.csv"

# --- CIFAR-100 fine → coarse mapping (20 superclasses) ---
cifar100_coarse = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet pepper"],
    "household electrical devices": ["clock", "computer keyboard", "lamp", "telephone", "television", "lawn mower"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple", "oak", "palm", "pine", "willow"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup truck", "train"],
    "vehicles 2": ["rocket", "streetcar", "tank", "tractor"]
}

# Flatten CIFAR fine→coarse for lookup
fine_to_coarse = {}
for coarse, fines in cifar100_coarse.items():
    for f in fines:
        fine_to_coarse[f.replace(" ", "_")] = coarse  # normalize

fine_labels = list(fine_to_coarse.keys())

# --- Load ImageNet JSON ---
with open(imagenet_json, "r") as f:
    imagenet_classes = json.load(f)

# --- Helper: WordNet mapping ---
def map_with_wordnet(synset_id):
    try:
        wn_syn = wn.synset_from_pos_and_offset("n", int(synset_id[1:]))
    except:
        return None
    for path in wn_syn.hypernym_paths():
        for h in path:
            lemma = h.lemma_names()[0].lower().replace(" ", "_")
            if lemma in fine_to_coarse:
                return fine_to_coarse[lemma]
    return None

# --- Helper: Fallback similarity mapping ---
def map_with_similarity(name):
    doc1 = nlp(name.replace("_", " "))
    best_score, best_coarse, best_fine = -1, None, None
    for fine, coarse in fine_to_coarse.items():
        score = doc1.similarity(nlp(fine.replace("_", " ")))
        if score > best_score:
            best_score, best_coarse, best_fine = score, coarse, fine
    return best_coarse, best_score, best_fine

# --- Build mapping ---
mapping = []
for idx, (wnid, name) in imagenet_classes.items():
    cifar_super, method, sim_score, fine_match = None, None, None, None

    # 1. Try WordNet mapping
    wn_match = map_with_wordnet(wnid)
    if wn_match:
        cifar_super = wn_match
        method = "wordnet"
        fine_match = "via_wordnet"  # marker for source
    else:
        # 2. Fallback: similarity
        cifar_super, sim_score, fine_match = map_with_similarity(name)
        method = "similarity"

    mapping.append({
        "ImageNet_ID": wnid,
        "ImageNet_Class": name,
        "CIFAR_Superclass": cifar_super,
        "MappingMethod": method,
        "SimilarityScore": sim_score if sim_score else "",
        "Closest_FineLabel": fine_match if fine_match else ""
    })

# --- Save CSV ---
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "ImageNet_ID", "ImageNet_Class", "CIFAR_Superclass"
    ])
    writer.writeheader()
    for row in mapping:
        writer.writerow({
            "ImageNet_ID": row["ImageNet_ID"],
            "ImageNet_Class": row["ImageNet_Class"],
            "CIFAR_Superclass": row["CIFAR_Superclass"]
        })

print(f"✅ Mapping saved (3 columns only) to {output_csv}")

