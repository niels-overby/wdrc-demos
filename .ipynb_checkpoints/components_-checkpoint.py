from lyd import generation, reverberation, wdrc, enhancement, utils, processing
import json

try:
    with open('../../datapaths.json') as f:
        audiodata = json.load(f)

except:
    with open('datapaths.json') as f:
        audiodata = json.load(f)



class processor_creator():
    def __init__(self,fs,nr_=None,wdrc_=None):
        self.fs = fs
        self.nr = nr_
        self.wdrc = wdrc_
        self.thr = 50
        self.ratio = 2
        self.wdrc_choice = 'fast'
        self.nr_choice = 'none'

        self.update_dicts()

    def update_param(self,param,val):
        if param == 'thr':
            self.thr = float(val)
            print(self.thr)
        elif param == 'ratio':
            self.ratio = float(val)
        elif param == 'wdrc':
            self.wdrc = self.wdrc_dict[val]
            self.wdrc_choice = val
        elif param =='nr':
            self.nr = self.nr_dict[val]
            self.nr_choice = val
        elif param =='apply':
            self.gen_sys()
        None
    
    def update_dicts(self):
        self.wdrc_dict = {
            'Fast Acting' : wdrc.WDRC(atk=5e-3,rel=50e-3,ratio=self.ratio,thr=self.thr,fs=self.fs),
            'Slow Acting' : wdrc.WDRC(atk=5e-3,rel=2000e-3,ratio=self.ratio,thr=self.thr,fs=self.fs),
            'Aware' : wdrc.WDRC(atk=5e-3,rel=[50e-3,2000e-3],ratio=self.ratio,thr=self.thr,fs=self.fs),
        }

        self.nr_dict = {
            'None': False,
            'Mild' : enhancement.LogMMSE(fs=self.fs,alpha=0.98,gain_min=utils.from_dB(-12)),
            'Moderate' : enhancement.LogMMSE(fs=self.fs,alpha=0.98,gain_min=utils.from_dB(-24)),
            None : False
        }

    def gen_sys(self):

        self.update_dicts()
        self.wdrc = self.wdrc_dict[self.wdrc_choice]
        self.nr = self.nr_dict[self.nr_choice]
        
        print(self.wdrc.rel)

        if self.nr is False:
            self.s =  self.wdrc
        else:
            self.s =  wdrc.NR_WDRC(self.nr,self.wdrc)

class scene_creator():
    def __init__(self,n_concats=10):
        fs = 16e3
        self.speech = generation.TIMIT(audiodata['timit'],fs=fs)
        self.n_concats = n_concats
        ## initialize noise dict
        n_ = [generation.FindSound(
                audiodata['data'],
                target_sound = noi,fs=fs) for noi in ['icra_01','icra_04']]

        self.n_dict = dict(zip(['Stationary','Modulated'],n_))

        ## Initialize room dict
        rooms = ['Anechoic','A','B','C','D']
        r_ = [reverberation.Surrey(audiodata['surrey'],
                            direct_sec=2e-3,fs=fs,
                            room=r,azimuth=0) for r in rooms]
        self.r_dict = dict(zip(rooms,r_))
        
        self.snr = 12
        
    def update_param(self,param,val):
        if param == 'signal_level':
            #print("Updated sig level")
            self.siglevel_param = float(val)

        elif param == 'snr':
            #print("Updated snr")
            self.snr_param = float(val)

        elif param == 'noise_type':
            #print("Updated noise type")
            self.noise_param = val
            self.noise = self.n_dict[val]

        elif param == 'room':
            #print("Updated room")
            self.room_param = val
            self.room = self.r_dict[val]

        elif param == 'apply':
            self.gen_scene()

        
    def gen_scene(self):
        self.stim = generation.NoisyReverberantMixture(self.speech,self.noise,self.room,SNR=self.snr_param,
                                            pre_noise=0.5,post_noise=0.5,
                                            n_concats=self.n_concats)
        
        

scene_gen = scene_creator()