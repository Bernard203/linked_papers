import scipy.sparse as sp
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
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]

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
    return X_enhanced_train, X_enhanced_val, X_enhanced_test, Y_train, Y_val, Y_test


def train_svm(x_train, y_train, x_val, y_val, n_splits=6):
    """
    训练 SVM 分类器
    """
    # print("Standardizing features...")
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train)
    x_train_splits = np.array_split(x_train, n_splits)
    y_train_splits = np.array_split(y_train, n_splits)
    models = []
    for i in tqdm(range(n_splits)):
        clf = SVC(kernel='rbf', gamma='auto', random_state=42)
        clf.fit(x_train_splits[i], y_train_splits[i])
        models.append(clf)
    print("Testing SVM classifier...")
    acc = 0

    # x_val = scaler.transform(x_val)
    for i,(x, y) in tqdm(enumerate(zip(x_val, y_val))):
        x = x.reshape(1, -1)
        y_pred = []
        for model in models:
            y_pred.append(model.predict(x)[0])
        y_pred = np.array(y_pred)
        y_pred = np.argmax(np.bincount(y_pred))
        if y_pred == y:
            acc += 1
        if i%1000 == 1:
            print(f"Accuracy: {acc/i:.4f}")
    acc /= len(y_val)
    print(f"Validation Accuracy: {acc:.4f}")

    return models


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
    x_train, x_val, x_test, y_train, y_val, y_test = process_data()
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_val, y_val = shuffle(x_val, y_val, random_state=42)
    print("Training SVM classifier...")
    train_svm(x_train, y_train, x_val, y_val)