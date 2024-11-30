from django.db import models
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.core.paginator import Paginator
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import AnonymousUser
import pickle
import os


class Essay(models.Model):
    title = models.CharField(max_length=200)
    abstract = models.TextField()
    year = models.IntegerField()
    category = models.CharField(max_length=100, null=True, blank=True)  # 论文标签
    feature_vector = models.BinaryField()  # 存储128维特征向量的二进制格式

    @staticmethod
    def load_feature_vectors():
        """加载所有论文的特征向量及对应标签"""
        essays = Essay.objects.filter(year__lte=2018)  # 训练集和验证集
        features = []
        labels = []
        for essay in essays:
            feature_vector = np.frombuffer(essay.feature_vector, dtype=np.float32)
            features.append(feature_vector)
            labels.append(essay.category)
        return np.array(features), np.array(labels)

    @staticmethod
    def train_classifier():
        """训练分类器并保存模型"""
        features, labels = Essay.load_feature_vectors()
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(features, labels)

        # 保存模型
        with open('./classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        print("Finished and saved！")

    @staticmethod
    def predict_category(feature_vector):
        """预测单篇论文的类别"""
        if not os.path.exists('./classifier.pkl'):
            raise FileNotFoundError("please train the model first！")

        # 加载模型
        with open('./classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)

        # 预测类别
        feature_vector = np.array(feature_vector).reshape(1, -1)
        predicted_category = classifier.predict(feature_vector)
        return predicted_category[0]

    @staticmethod
    def classify_test_set():
        """对测试集的论文进行分类"""
        essays = Essay.objects.filter(year__gte=2019)  # 测试集
        if not os.path.exists('./classifier.pkl'):
            raise FileNotFoundError("please train the model first！")

        # 加载模型
        with open('./classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)

        for essay in essays:
            feature_vector = np.frombuffer(essay.feature_vector, dtype=np.float32)
            essay.category = classifier.predict([feature_vector])[0]
            essay.save()  # 保存预测结果
        print("Test set has been trained！")

    @staticmethod
    def search_by_keyword(keyword, page, per_page=10):
        """分页搜索论文"""
        essays = Essay.objects.filter(title__icontains=keyword)
        paginator = Paginator(essays, per_page)
        try:
            return paginator.page(page)
        except:
            return paginator.page(paginator.num_pages)

    @staticmethod
    def find_similar_papers(essay_id, top_n=5):
        """找到与指定论文最相似的论文"""
        target_essay = Essay.objects.get(id=essay_id)
        target_vector = np.frombuffer(target_essay.feature_vector, dtype=np.float32)

        all_essays = Essay.objects.exclude(id=essay_id)
        similarities = []
        for essay in all_essays:
            feature_vector = np.frombuffer(essay.feature_vector, dtype=np.float32)
            similarity = np.dot(target_vector, feature_vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(feature_vector)
            )
            similarities.append((essay, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [essay for essay, _ in similarities[:top_n]]

class User(AbstractUser):
    # email = models.EmailField(unique=True)
    nickname = models.CharField(max_length=50, blank=True, null=True)
    identity = models.CharField(max_length=255, blank=True, null=True)
    role = models.CharField(max_length=255, blank=True, null=True)


    def save(self, *args, **kwargs):
        if not self.password.startswith('pbkdf2_'):  # 避免重复加密
            self.password = make_password(self.password)
        super().save(*args, **kwargs)

# class CustomAnonymousUser(AnonymousUser):
#     nickname = None
#     identity = None
#     role = None

class Edge(models.Model):
    essay = models.ForeignKey(Essay, related_name='citing', on_delete=models.CASCADE)
    cited = models.ForeignKey(Essay, related_name='cited_by', on_delete=models.CASCADE)
