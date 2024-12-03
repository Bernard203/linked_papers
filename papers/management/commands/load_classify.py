from django.core.management.base import BaseCommand
from papers.utils.classify import classify

class Command(BaseCommand):
    help = 'Train and evaluate SVM classifier'

    def handle(self, *args, **kwargs):
        classify()
        self.stdout.write(self.style.SUCCESS('Successfully ran classify function'))