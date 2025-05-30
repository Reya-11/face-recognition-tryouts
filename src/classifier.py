from sklearn.svm import SVC
import pickle

def train_classifier(X, y, save_path="classifier.pkl"):
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

def load_classifier(path="classifier.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
