from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def classification(type, X_data, labels):
    
    #Split the Dataset into training and test set (60% Training / 40% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_data, labels, test_size=0.4, random_state=42)

    if (type=="KNN"):
        # ## KNN Classification
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, y_train)
        print("KNN accuracy :", neigh.score(X_test, y_test, sample_weight=None))
        print("Confusion matrix :\n", confusion_matrix(y_test, neigh.predict(X_test)),"\n")

    elif (type=="SVM"):
        # ## SVM Classification
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        print("SVM accuracy :", clf.score(X_test, y_test,sample_weight=None))
        print("Confusion matrix :\n", confusion_matrix(y_test, clf.predict(X_test)),"\n")

    return 0