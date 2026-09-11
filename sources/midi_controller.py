import mido
import os,sys


class LPD8:
    def __init__(self,name_midi='LPD8'):
        try:
            name = self.search_lpd8(name_midi)
        except:
            print('device ',name_midi,' not found, stopping')
            sys.exit()
        self.name_midi_controller = name
        print('found device = ',self.name_midi_controller)
        self.port = mido.open_ioport(self.name_midi_controller)

    def search_lpd8(self,name_midi):
        
        names = mido.get_ioport_names()
        names = set(n for n in names if name_midi in n)
        assert len(names) == 1
        return names.pop()