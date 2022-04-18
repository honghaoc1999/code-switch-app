"""codeswitch URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from voice2text import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.recorderView, name='recorderView'),
    path('6s_retry_recorder', views.periodic6sRetryRecorder, name='6s_retry_recorder'),
    path('3s_retry_recorder', views.periodicRetryRecorder, name='3s_retry_recorder'),
    path('entire_retry', views.entireRetryRecorder, name='entire_retry'),
    path('silent_chunking', views.silentChunkRecorder, name='silent_chunking'),
    path('transcribeAudio', views.transcribeAudio, name='transcribeAudio')
]
