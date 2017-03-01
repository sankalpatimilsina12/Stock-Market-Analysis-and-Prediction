from django.db import models
from django.utils import timezone
import csv
import pandas as pd
# Create your models here.


class Post(models.Model):
    author = models.ForeignKey('auth.User')
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(
        default=timezone.now)
    published_date = models.DateTimeField(
        blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title


class stock_names(models.Model):
    def get_all_names():
        fp=open('/home/sankalpa/Major_project/Stock_Analysis/my_file.csv')
        r=csv.reader(fp, delimiter=',')
        item=[]
        for row in r:
            temp=[]
            for col in row:
                temp.append(col)
            item.append(temp)
        return item

class stocks_data(models.Model):
    def get_data(stock_name):
        fp = open('/home/sankalpa/simple-blog/mysite/blog/templates/blog/stocks-data/' + stock_name + '.csv')
        r = csv.reader(fp, delimiter = ',')
        data = []
        for row in r:
            temp = []
            for col in row:
                temp.append(col)
            data.append(temp)
        return data
