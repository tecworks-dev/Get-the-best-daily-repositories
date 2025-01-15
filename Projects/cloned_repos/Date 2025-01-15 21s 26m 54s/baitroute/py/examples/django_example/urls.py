from django.urls import path
from django.http import HttpResponse

def home(request):
    return HttpResponse('Welcome to the real application!')

urlpatterns = [
    path('', home, name='home'),
] 