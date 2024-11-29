from django.core.management.base import BaseCommand
from django.db import connection

from papers.data_loader import load_essays_into_db, load_edges_into_db

class Command(BaseCommand):
    help = "Load papers and citation dataset into the database"

    def handle(self, *args, **kwargs):
        # # Papers
        # self.stdout.write(self.style.SUCCESS("Loading papers into the database..."))
        # try:
        #     load_essays_into_db()
        #     self.stdout.write(self.style.SUCCESS("Papers loaded successfully!"))
        # except Exception as e:
        #     self.stdout.write(self.style.ERROR(f"Error loading papers: {e}"))

        # Edges
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF;")
        self.stdout.write(self.style.SUCCESS("Loading citation edges into the database..."))
        try:
            load_edges_into_db()
            self.stdout.write(self.style.SUCCESS("Citation edges loaded successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading citation edges: {e}"))
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = ON;")
