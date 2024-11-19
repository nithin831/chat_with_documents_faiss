from django.contrib import admin
from django.urls import path
from . import views as v

urlpatterns = [
    path('chat/', v.chatbot, name = "chatbot"),
    path('', v.doc, name = "doc"),
    
]