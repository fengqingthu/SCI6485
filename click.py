from pynput.mouse import Button, Controller
import time
import random

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    time.sleep(random.randint(30, 90))
