import time
import pyautogui

# Set the duration for 5 minutes (300 seconds)
end_time = time.time() + 300  

pyautogui.click(1178, 1056)  # Click first location
while time.time() < end_time:
    pyautogui.click(570, 446)    # Click second location
    time.sleep(1)  # Wait for UI to update

    pyautogui.press('down')  # Press down key to select the option
    time.sleep(1)
    pyautogui.press('enter')

    pyautogui.click(1652, 525)  # Click third location

    time.sleep(1)  # Add delay between loops to avoid excessive clicking

print("Clicking stopped after 5 minutes.")
