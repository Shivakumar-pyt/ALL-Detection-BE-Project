from django.urls import path
from .views import main_page, process_data

urlpatterns = [
    path('',main_page, name="main_page"),
    path('processData',process_data, name="process_data")
]