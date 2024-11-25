import PySimpleGUI as sg
import subprocess

# GUIのレイアウト
layout = [
    [sg.Text('Select a mode for analysis', size=(30, 1), justification='center', font=("Helvetica", 16))],
    [sg.Button('Impose Right', size=(15, 2), button_color=('white', 'gray')), 
     sg.Button('Impose Left', size=(15, 2), button_color=('white', 'blue'))],
    [sg.Button('Trace Right', size=(15, 2), button_color=('white', 'darkgray')), 
     sg.Button('Trace Left', size=(15, 2), button_color=('white', 'black'))],
    [sg.Button('Exit', size=(32, 2), button_color=('white', 'lightgray'))]
]

# ウィンドウの作成
window = sg.Window('Throwing Analysis', layout, size=(400, 250), element_justification='center', background_color='lightgrey')

# イベントループ
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':  # ウィンドウの×ボタンまたはExitボタンがクリックされたとき
        break
    elif event == 'Impose Right':  # Impose Rightボタンがクリックされたとき
        subprocess.run(['python', 'detect_lgbm_impose_rt.py'])
    elif event == 'Impose Left':  # Impose Leftボタンがクリックされたとき
        subprocess.run(['python', 'detect_lgbm_impose_lt.py'])
    elif event == 'Trace Right':  # Trace Rightボタンがクリックされたとき
        subprocess.run(['python', 'trace_rt.py'])
    elif event == 'Trace Left':  # Trace Leftボタンがクリックされたとき
        subprocess.run(['python', 'trace_lt.py'])

window.close()


