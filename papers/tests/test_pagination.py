from django.test import TestCase
from ..models import Essay
from ..pagination import fetch

class PaginationTests(TestCase):
    def setUp(self):
        # 创建测试论文
        category = "TestCategory"
        for i in range(30):
            Essay.objects.create(title=f"Test Paper {i}", abstract="Abstract", year=2021, category=category)
        self.target_essay = Essay.objects.first()

    def test_fetch_related_essays(self):
        # 获取第一页数据
        result = fetch(essayId=self.target_essay.id, keyword=None, page=1)
        self.assertEqual(len(result['essays']), 16)  # 每页16篇
        self.assertEqual(result['current_page'], 1)
        self.assertGreaterEqual(result['total_pages'], 2)  # 至少2页
