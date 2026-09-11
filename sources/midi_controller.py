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
        self.pad_map = {36:0,37:1,38:2,39:3,40:4,41:5,42:6,43:7}

    def search_lpd8(self,name_midi):
        names = mido.get_ioport_names()
        names = set(n for n in names if name_midi in n)
        assert len(names) == 1
        return names.pop()

    def init_state(self):
        self.knobs = np.zeros(8,int)
        self.padOnOff = np.zeros(8,bool)
        self.padVelo = np.zeros(8,int)
    
    def set_knob(ind,val):
        self.knobs[ind-1] = val

    def set_pad(note,OnOff,velo):
        pass