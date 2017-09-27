from django.conf.urls import url

from . import views

app_name = 'dataset'
urlpatterns = [
    url(r'^$', views.ImagesView.as_view(), name='image_list'),
    # url(r'^(?P<trainset>[1-2]+)$', views.ImagesView.as_view(), name='trainset'),
]