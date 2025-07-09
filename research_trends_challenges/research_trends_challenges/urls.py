"""research_trends_challenges URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
"""

from django.contrib import admin
from django.urls import re_path
from Remote_User import views as remoteuser
from research_trends_challenges import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static

urlpatterns = [
    re_path(r'^admin/', admin.site.urls),
    re_path(r'^$', remoteuser.index, name="index"),
    re_path(r'^login/$', remoteuser.login, name="login"),
    re_path(r'^Register1/$', remoteuser.Register1, name="Register1"),
    re_path(r'^Predict_Cyber_Attack_Status/$', remoteuser.Predict_Cyber_Attack_Status, name="Predict_Cyber_Attack_Status"),
    re_path(r'^ViewYourProfile/$', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    re_path(r'^serviceproviderlogin/$', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    re_path(r'^View_Remote_Users/$', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    re_path(r'^charts/(?P<chart_type>\w+)', serviceprovider.charts, name="charts"),
    re_path(r'^charts1/(?P<chart_type>\w+)', serviceprovider.charts1, name="charts1"),
    re_path(r'^likeschart/(?P<like_chart>\w+)', serviceprovider.likeschart, name="likeschart"),
    re_path(r'^View_Cyber_Attack_Type_Ratio/$', serviceprovider.View_Cyber_Attack_Type_Ratio, name="View_Cyber_Attack_Type_Ratio"),
    re_path(r'^train_model/$', serviceprovider.train_model, name="train_model"),
    re_path(r'^View_Prediction_Of_Cyber_Attack_Status/$', serviceprovider.View_Prediction_Of_Cyber_Attack_Status, name="View_Prediction_Of_Cyber_Attack_Status"),
    re_path(r'^Download_Predicted_DataSets/$', serviceprovider.Download_Predicted_DataSets, name="Download_Predicted_DataSets"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
