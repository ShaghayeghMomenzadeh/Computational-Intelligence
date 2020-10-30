import os

import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score as error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# This function is created for some pre-processing usage such as creating suitable dataset
def pre_process(size=50):
    DATA = []
    Labels = []
    for path, _, files in os.walk("persian_digit"):
        for name in files:
            imagePath = os.path.join(path, name)
            # Read image as gray scale
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            # Resize image for get better feature
            resizeImg = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            # Thredshold for 128 is 0 and 255 is
            _, IMG = cv2.threshold(resizeImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Append extract featured DATA
            DATA.append(pca(IMG))
            # Label of each image
            Labels.append(path[14:])

    return DATA, Labels


# In this function, we use PCA algorithm to extract the top 5 important features from input image
def pca(data_in):
    model = PCA(n_components=10)
    model.fit(data_in)
    X = model.transform(data_in)
    flatten_feature = list(X.flatten())

    return flatten_feature


# This is our model. We use MLP in order to predict the persian digits
def mlp(x_train, x_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=5, max_iter=2000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("#########################################################")
    print("#########  Accuracy: ", 1 - error(y_test, y_pred), "   ##########")
    print("#########################################################")


def main():
    # Size of resizing image
    size = 50
    # Extract feature of image
    DATA, Labels = pre_process(size)
    # Set train and test data
    X_train, X_test, y_train, y_test = train_test_split(DATA, Labels, test_size=0.10)
    mlp(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
