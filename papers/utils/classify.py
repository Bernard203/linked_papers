import scipy.sparse as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import numpy as np

from ..data_loader import load_features, load_edges, load_papers
from tqdm import tqdm

from ..models import Essay


def process_data():
    """
    处理数据，包括特征矩阵、邻接矩阵、标签、数据增强等步骤。
    """
    print("Processing data...")
    features = load_features()
    src, dst = load_edges()
    papers_df = load_papers()

    # Feature matrix
    X = features
    n, d = features.shape
    print("n, d:", n, d)

    # Adjacency matrix
    A = sp.coo_matrix((np.ones(src.shape[0]), (src, dst)), shape=(n, n), dtype=np.float32)

    # Build symmetric adjacency matrix
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)  # Ensures symmetry
    # npA = A.toarray()

    # Add self-loops
    A = A + sp.eye(n, dtype=np.float32)

    # Labels (one-hot encoding)
    papers_df['year'] = papers_df['year'].astype(int)
    train_mask = papers_df['year'] <= 2017
    val_mask = papers_df['year'] == 2018
    test_mask = papers_df['year'] >= 2019

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(papers_df['category'])

    # Splits: Get training, validation, and test indices
    idx_train = train_mask[train_mask].index.tolist()
    idx_val = val_mask[val_mask].index.tolist()
    idx_test = test_mask[test_mask].index.tolist()

    # Split data
    # X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    Y_train, Y_val= Y[idx_train], Y[idx_val]

    # Step 1: Compute the degree matrix D
    D = sp.diags(np.array(A.sum(axis=1)).flatten(), dtype=np.float32)

    # Step 2: Compute the enhanced feature matrix D^(-1) * A * X
    D_inv = sp.diags(1.0 / D.diagonal())
    X_enhanced = D_inv.dot(A).dot(X)

    # Convert to dense matrix if necessary
    X_enhanced = X_enhanced.toarray() if sp.issparse(X_enhanced) else X_enhanced

    # Split enhanced features
    X_enhanced_train, X_enhanced_val, X_enhanced_test = X_enhanced[idx_train], X_enhanced[idx_val], X_enhanced[idx_test]

    print("Data processed!")
    return X_enhanced_train, X_enhanced_val, X_enhanced_test, Y_train, Y_val, idx_test, label_encoder


def train_ensemble(x_train, y_train, x_val, y_val, n_splits=6):
    """
    训练 SVM 分类器
    """
    # print("Standardizing features...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_train_splits = np.array_split(x_train_scaled, n_splits)
    y_train_splits = np.array_split(y_train, n_splits)
    svm_models = []
    dt_models = []
    vote_models = []

    for i in tqdm(range(n_splits)):
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(x_train_splits[i], y_train_splits[i])
        svm_models.append(clf)

        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_clf.fit(x_train_splits[i], y_train_splits[i])
        dt_models.append(dt_clf)

        voting_clf = VotingClassifier(estimators=[('svm', clf), ('dt', dt_clf)], voting='soft')
        voting_clf.fit(x_train_splits[i], y_train_splits[i])
        vote_models.append(voting_clf)

    print("Testing classifier...")
    acc = 0
    x_val = scaler.transform(x_val)

    for i,(x, y) in tqdm(enumerate(zip(x_val, y_val))):
        x = x.reshape(1, -1)
        y_pred = []
        for svm_model, dt_model, vote_model in zip(svm_models, dt_models, vote_models):
            svm_pred = svm_model.predict(x)[0]
            dt_pred = dt_model.predict(x)[0]
            vt_pred = vote_model.predict(x)[0]
            y_pred.append(svm_pred)
            y_pred.append(dt_pred)
            y_pred.append(vt_pred)
        y_pred = np.array(y_pred)
        y_pred = np.argmax(np.bincount(y_pred))
        if y_pred == y:
            acc += 1
        if i%1000 == 1:
            print(f"Accuracy: {acc/i:.4f}")
    acc /= len(y_val)
    print(f"Validation Accuracy: {acc:.4f}")

    # voting_clf = VotingClassifier(estimators=[('svm', svm_models), ('dt', dt_models)], voting='soft')

    return svm_models, dt_models, vote_models


def evaluate_svm(clf, x_test, y_test):
    """
    使用训练好的 SVM 模型评估测试集的性能。
    """
    y_test_pred = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

def classify(n_splits=6):
    """
    训练和评估 SVM 分类器
    """
    x_train, x_val, x_test, y_train, y_val, y_test, le = process_data()
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_val, y_val = shuffle(x_val, y_val, random_state=42)
    print("Training SVM classifier...")
    # train_ensemble(x_train, y_train, x_val, y_val)
    svm_models, dt_models, vote_models = train_ensemble(x_train, y_train, x_val, y_val, n_splits=n_splits)

    essays = Essay.objects.filter(id__in=y_test)

    for essay in essays:
        feature_vector = np.frombuffer(essay.feature_vector, dtype=np.float32).reshape(1, -1)
        y_pred = []
        for svm_model, dt_model, vote_model in zip(svm_models, dt_models, vote_models):
            svm_pred = svm_model.predict([feature_vector])[0]
            dt_pred = dt_model.predict([feature_vector])[0]
            vt_pred = vote_model.predict([feature_vector])[0]
            y_pred.append(svm_pred)
            y_pred.append(dt_pred)
            y_pred.append(vt_pred)
        y_pred = np.array(y_pred)
        y_pred = np.argmax(np.bincount(y_pred))
        original_labels = le.inverse_transform(y_pred)
        essay.category = original_labels
        essay.save()