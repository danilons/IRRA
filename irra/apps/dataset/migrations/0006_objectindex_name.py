# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-10-07 14:15
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0005_auto_20171007_1257'),
    ]

    operations = [
        migrations.AddField(
            model_name='objectindex',
            name='name',
            field=models.CharField(default='', max_length=30),
            preserve_default=False,
        ),
    ]