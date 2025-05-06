from django.db import models
from .utils import decrypt

class Keystroke(models.Model):
    key = models.CharField(max_length=10)  # Store each keypress
    # press_time = models.FloatField(null=True, blank=True)  # Timestamp when the key is pressed
    # release_time = models.FloatField(null=True, blank=True)  # Timestamp when the key is released
    hold_time = models.FloatField(null=True, blank=True)  # Time key was held down (release - press)
    flight_time = models.FloatField(null=True, blank=True)  # Time between release of previous key and press of current key
    dd_time = models.FloatField(null=True, blank=True)  # Time between two consecutive key presses (Down-Down)
    # ud_time = models.FloatField(null=True, blank=True)  # Time between key release and next key press (Up-Down)
    # user_ip = models.GenericIPAddressField()
    
    # def get_decrypted_data(self):
    #     return {
    #         "key": decrypt(self.key),
    #         "press_time": decrypt(self.press_time),
    #         "release_time": decrypt(self.release_time) if self.release_time else None,
    #         "hold_time": decrypt(self.hold_time) if self.hold_time else None,
    #         "user_ip": decrypt(self.user_ip),
    #         "dd_time": decrypt(self.dd_time) if self.dd_time else None,
    #         "flight_time": decrypt(self.flight_time) if self.flight_time else None
    #     }
    

    def __str__(self):
        return f"Key: {self.key} - DD: {self.dd_time}, FT: {self.flight_time}, HT: {self.hold_time}"