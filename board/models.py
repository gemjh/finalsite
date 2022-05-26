from django.db import models

# Create your models here.
class Board(models.Model):
    title = models.CharField("제목", max_length=255)
    content = models.TextField("본문")