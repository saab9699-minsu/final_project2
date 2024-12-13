from django.shortcuts import render


# Create your views here.

def about(request):
    return render(request, "about.html")

def contact(request):
    return render(request, "contact.html")

def do(request):
    return render(request, "do.html")

def index(request):
    return render(request, "index.html")

def portfolio(request):
    return render(request, "portfolio.html")