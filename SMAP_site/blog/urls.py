from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^index.html$', views.index, name='index'),
    url(r'^listed_companies.html$', views.listed_companies, name='listed_companies'),
    url(r'^analysis.html$', views.analysis, name='analysis'),
    url(r'^projects.html$', views.projects, name='projects'),
    url(r'^contact.html$', views.contact, name='contact'),
    url(r'^charts/simple.png$', views.simple),
    url(r'^stocks/(?P<stock_name>\w+)/$', views.listing, name='listing'),
    url(r'^simple/(?P<stock_name>\w+)/(?P<plot_kind>\w+)/$', views.simple, name='simple'),
    url(r'^simple/(?P<stock_name>\w+)/$', views.simple, name='simple'),
    url(r'^train$', views.train, name='train'),


]
