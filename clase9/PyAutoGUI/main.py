import pyautogui, sys
print('Press Ctrl-C to quit.')
print(pyautogui.size())
#pyautogui.moveTo(0,0)
pyautogui.moveTo(97,11)
pyautogui.click()

try:
    while True:
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
except KeyboardInterrupt:
    print('\n')
