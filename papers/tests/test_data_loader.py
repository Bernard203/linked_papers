from django.test import TestCase
from papers.data_loader import load_papers, load_features, load_edges

class DataLoaderTests(TestCase):

    def test_load_papers(self):
        """测试是否成功加载论文数据"""
        papers = load_papers()
        self.assertIsNotNone(papers)  # 检查是否加载成功
        self.assertGreater(len(papers), 0)  # 确保有数据
        self.assertIn('title', papers.columns)  # 检查关键列是否存在
        self.assertIn('year', papers.columns)

    def test_load_features(self):
        """测试是否成功加载特征向量"""
        features = load_features()
        self.assertIsNotNone(features)  # 检查是否加载成功
        self.assertEqual(features.shape[1], 128)  # 确保特征向量为128维
        self.assertGreater(features.shape[0], 0)  # 确保有数据

    def test_load_edges(self):
        """测试是否成功加载引用关系"""
        citer, citee = load_edges()
        self.assertIsNotNone(citer)  # 检查是否加载成功
        self.assertIsNotNone(citee)
        self.assertEqual(len(citer), len(citee))  # 确保引用关系对齐
        self.assertGreater(len(citer), 0)  # 确保有数据
