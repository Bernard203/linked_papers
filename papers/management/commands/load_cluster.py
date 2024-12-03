from django.core.management.base import BaseCommand
from papers.utils.cluster import cluster

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS("Clustering by feature vectors..."))
        cluster()
        self.stdout.write(self.style.SUCCESS('Successfully clustered all papers!'))