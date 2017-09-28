from django.db import models


class Image(models.Model):
    TRAIN_SET = ((1, 'Train'), (2, 'Test'), (3, 'Valid'))
    filename = models.CharField(max_length=200)
    index = models.IntegerField()
    trainset = models.CharField(max_length=10, choices=TRAIN_SET)

    class Meta:
        ordering = ['index']


class Object(models.Model):
    name = models.CharField(max_length=30)
    db_index = models.IntegerField()
    index = models.IntegerField()
    r = models.IntegerField()
    g = models.IntegerField()
    b = models.IntegerField()

    class Meta:
        ordering = ['db_index']
        unique_together = (('r', 'g', 'b'),)


class Experiment(models.Model):
    TESTS = ((1, 'Lan(2012)'), (2, 'Malinowski(2014)'))
    exp = models.CharField(max_length=20, choices=TESTS)


class ImageObject(models.Model):
    image = models.ForeignKey(Image)
    object = models.ForeignKey(Object)
    experiment = models.ForeignKey(Experiment)

    x = models.TextField()
    y = models.TextField()

