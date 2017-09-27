from django.db import models


class Image(models.Model):
    TRAIN_SET = ((1, 'Train'), (2, 'Test'), (3, 'Valid'))
    filename = models.CharField(max_length=200)
    index = models.IntegerField()
    trainset = models.CharField(max_length=1, choices=TRAIN_SET)

    class Meta:
        ordering = ['index']


class Object(models.Model):
    name = models.CharField(max_length=30)
    index = models.IntegerField()

class ImageObject(models.Model):
    image = models.ForeignKey(Image)
    object = models.ForeignKey(Object)

    x = models.TextField()
    y = models.TextField()


