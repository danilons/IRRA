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
    index = models.IntegerField()
    r = models.IntegerField()
    g = models.IntegerField()
    b = models.IntegerField()

    class Meta:
        ordering = ['index']
        unique_together = (('r', 'g', 'b'),)


class ObjectIndex(models.Model):
    object = models.ForeignKey(Object)
    name = models.CharField(max_length=30)
    alias = models.CharField(max_length=30)
    index = models.IntegerField()


class Experiment(models.Model):
    TESTS = ((1, 'Lan(2012)'), (2, 'Malinowski(2014)'))
    exp = models.CharField(max_length=20, choices=TESTS)


class ImageObject(models.Model):
    image = models.ForeignKey(Image)
    object = models.ForeignKey(Object)
    experiment = models.ForeignKey(Experiment)

    x = models.TextField()
    y = models.TextField()


class Preposition(models.Model):
    name = models.CharField(max_length=10)
    index = models.IntegerField()

    class Meta:
        ordering = ['index']
        unique_together = (('name', 'index'), )


class Query(models.Model):
    query_type = models.CharField(max_length=1)
    query_name = models.CharField(max_length=50)
    trainset = models.CharField(max_length=1, choices=Image.TRAIN_SET)

    class Meta:
        unique_together = (('query_type', 'query_name'), )


class StructuredQueryDataset(models.Model):
    query = models.ForeignKey(Query)
    object1 = models.ForeignKey(Object, related_name='object1')
    object2 = models.ForeignKey(Object, related_name='object2')
    preposition = models.ForeignKey(Preposition)


class QueryNoun(models.Model):
    query = models.ForeignKey(Query)
    object = models.ForeignKey(Object)


class QueryGroundTruth(models.Model):
    query = models.ForeignKey(Query)
    image = models.ForeignKey(Image)


