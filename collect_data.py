#!/usr/bin/env python3
 
from pynput import keyboard
import numpy as np
import pandas as pd
import time

timestamps = {}
raw_data = []
start_time = 0
started = False
chars_pressed = 0

def extract_features(key_name, release_time):
    stored_vals = timestamps.pop(key_name)
    press_time = stored_vals[0]
    cpm = stored_vals[1]
    hold_time = release_time - press_time
    
    return (key_name, press_time, release_time, hold_time, cpm)


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:
        return str(key)
 
def on_press(key):
    global start_time
    global started
    global chars_pressed

    if(not started):
        start_time = time.time()
        started = True

    press_time = time.time()
    key_name = get_key_name(key)

    chars_pressed += 1

    cpm = chars_pressed/(press_time - start_time)*60

    print('Key {} pressed.'.format(key_name))

    if key_name != 'Key.esc':
        timestamps[key_name] = [press_time, cpm]

 
def on_release(key):
    release_time = time.time()
    key_name = get_key_name(key)

    print('Key {} released.'.format(key_name))

    if key_name == 'Key.esc':
        print('Exiting...')
        return False

    try:
        key_info = extract_features(key_name, release_time)
        
    except KeyError:
        key_info = extract_features(key_name.upper(), release_time)

    raw_data.append(key_info)


with keyboard.Listener(
    on_press = on_press,
    on_release = on_release) as listener:
    print("Start typing..")
    listener.join()


raw_data_arr = np.array(raw_data)
df = pd.DataFrame(raw_data_arr, columns=['key', 'press_time', 'release_time', 'hold_time', 'cpm'])
df.to_csv("output.csv")
