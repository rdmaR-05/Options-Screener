import time
import pyautogui
end_time = time.time() + 300  
pyautogui.click(1178, 1056)  
while time.time() < end_time:
    pyautogui.click(570, 446)   
    time.sleep(1) 
    pyautogui.press('down') 
    time.sleep(1)
    pyautogui.press('enter')
    pyautogui.click(1652, 525)  
    time.sleep(1) 
