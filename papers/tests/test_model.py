from django.test import TestCase
from ..models import Essay


class EssayTests(TestCase):
    def setUp(self):
        for i in range(50):
            Essay.objects.create(title=f"Test Paper {i}", abstract="Abstract", year=2017, category="CS")

    def test_pagination(self):
        results = Essay.search_by_keyword("Test", page=1, per_page=10)
        self.assertEqual(len(results), 10)

    def test_similar_papers(self):
        similar_papers = Essay.find_similar_papers(1, top_n=5)
        self.assertEqual(len(similar_papers), 5)
