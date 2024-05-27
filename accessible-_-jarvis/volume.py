import pygame.mixer as mixer

mixer.init()

current_volume = mixer.get_master_volume()
print(f"Current volume: {current_volume}")

# Rest of the code for volume control...
