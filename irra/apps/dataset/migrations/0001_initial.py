# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-09-27 15:44
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Experiment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('exp', models.CharField(choices=[(1, 'Lan(2012)'), (2, 'Malinowski(2014)')], max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=200)),
                ('index', models.IntegerField()),
                ('trainset', models.CharField(choices=[(1, 'Train'), (2, 'Test'), (3, 'Valid')], max_length=10)),
            ],
            options={
                'ordering': ['index'],
            },
        ),
        migrations.CreateModel(
            name='ImageObject',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('x', models.TextField()),
                ('y', models.TextField()),
                ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dataset.Experiment')),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dataset.Image')),
            ],
        ),
        migrations.CreateModel(
            name='Object',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('db_index', models.IntegerField()),
                ('index', models.IntegerField()),
                ('r', models.IntegerField()),
                ('g', models.IntegerField()),
                ('b', models.IntegerField()),
            ],
            options={
                'ordering': ['index'],
            },
        ),
        migrations.AlterUniqueTogether(
            name='object',
            unique_together=set([('r', 'g', 'b')]),
        ),
        migrations.AddField(
            model_name='imageobject',
            name='object',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dataset.Object'),
        ),
    ]
