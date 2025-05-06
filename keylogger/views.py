from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import Keystroke
from .utils import encrypt, decrypt

last_keystroke = None  # Store the last key event globally

def home(request):
    return render(request, 'keylogger/index.html')

@csrf_exempt
def capture_keystrokes(request):
    global last_keystroke

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received data:", data)


            key = data.get('key')  # Encrypt the key
            event_type = data.get('event_type')
            
            # user_ip = request.META.get('REMOTE_ADDR', '0.0.0.0')

            if not key or not event_type:
                return JsonResponse({'status': 'error', 'message': 'Invalid data'}, status=400)

            if event_type == "press":
                dd_time = data.get('dd_time')
                flight_time = data.get('flight_time')

                keystroke = Keystroke.objects.create(
                    key=key,
                    # user_ip=user_ip,
                    dd_time=dd_time if dd_time is not None else 0,
                    flight_time=flight_time if flight_time is not None else 0
                )
                last_keystroke = keystroke  # Store last keystroke

            elif event_type == "release":
                hold_time = data.get("hold_time", 0)
                try:
                    keystroke = Keystroke.objects.filter(key=key, hold_time__isnull=True).latest('id')

                    keystroke.hold_time = hold_time if hold_time is not None else 0
                    keystroke.save()
                    last_keystroke = keystroke  # Update last keystroke

                except Keystroke.DoesNotExist:
                    return JsonResponse({'status': 'error', 'message': f'No matching keypress found for release {key}'}, status=400)


            return JsonResponse({'status': 'success'}, status=201)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)
