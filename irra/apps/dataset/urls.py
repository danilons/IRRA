from django.conf.urls import url

from . import views

app_name = 'dataset'
urlpatterns = [
    url(r'^$', views.ImageListView.as_view(), name='image_list'),
    url(r'^(?P<pk>[-\w]+)/$', views.ImageDetailView.as_view(), name='image_detail'),
    url(r'query_list$', views.QueryListView.as_view(), name='query_list'),
    url(r'query_detail/(?P<pk>[-\w]+)/$', views.QueryDetailView.as_view(), name='query_detail')
]
