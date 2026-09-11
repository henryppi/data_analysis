# from time import sleep
from sources.midi_controller import LPD8
# import mido


midi = LPD8()

# print(list(midi.port.iter_pending()))


for msg in midi.port:
    # Filter for control change or other messages
    if msg.type == 'control_change':
        print(f"Control: {msg.control}, Value: {msg.value}, Channel: {msg.channel}")
    else:
        print(msg)

