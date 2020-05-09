import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    return

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def find_eigenfaces(n,dataset):
    eigens=PCA(n_components=n,svd_solver='randomized',whiten=True).fit(dataset)
    return eigens

def SVM(grid,x,y):
    SVM_result=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),grid)
    SVM_result=SVM_result.fit(x,y)
    return SVM_result

def main():
    # Downloads the dataset from the sklearn library
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    n_features = X.shape[1]
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Dataset Size:")
    print("Number of Samples: %d" % n_samples)
    print("Number of Features: %d" % n_features)
    print("Number of Classes: %d" % n_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    n = 150
    print("Extracting the top %d eigenfaces from %d faces"% (n, X_train.shape[0]))

    pca = find_eigenfaces(n,X_train)

    eigenfaces = pca.components_.reshape((n, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

#-------------------------------SVM------------------
    print("Fitting the classifier to the training set")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = SVM(param_grid,X_train_pca,y_train)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    y_pred = clf.predict(X_test_pca)

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


    prediction_titles = [title(y_pred, y_test, target_names, i)
                        for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()
if __name__=="__main__":
    main()