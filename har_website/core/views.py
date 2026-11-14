from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request,"core/home.html")


def login_view(request):
    return render(request, "core/login.html")


def register_view(request):
    return render(request, "core/register.html")

def dashboard(request):
    return render(request, "core/dashboard.html")

def live_recognition(request):
    return render(request,"core/live.html")