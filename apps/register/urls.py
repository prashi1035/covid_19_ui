from django.conf.urls import url
from . import views
urlpatterns = [
    url(r'^$', views.index),
    url(r'^register$', views.register),
    url(r'^success$', views.success),
    url(r'^login$', views.login),
    url(r'^covid$', views.covid),
    url(r'^covid_predictions$', views.covid_predictions),
    url(r'^homepage$', views.homepage),
    url(r'^logout_req$', views.logout_req)

]