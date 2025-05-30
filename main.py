# Prepare training embeddings
# Assume preprocessed aligned images from LFW
import os
from src.embedder import get_embedding
from src.classifier import train_classifier
import cv2
from src.align_face import align_face

X, y = [], []
lfw_path = "assets/lfw_dataset/"

for person in os.listdir(lfw_path):
    person_path = os.path.join(lfw_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        aligned = align_face(img)
        if aligned is not None:
            emb = get_embedding(aligned)
            X.append(emb)
            y.append(person)

train_classifier(X, y)
