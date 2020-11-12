import numpy as np
import pandas as pd
import os
from scipy.signal import welch
import matplotlib.pyplot  as plt
import ipywidgets as wi
import warnings
import time

#warnings.filterwarnings('ignore')


#figure.figsize : 8, 6
#figure.autolayout : True

def config_plots():
    #plt.style.use(style)
    plt.rcParams.update({
      "axes.spines.top" : False,
      "axes.spines.right" : False,
      'font.size': 12,
      "xtick.major.size" : 5,
      "ytick.major.size" : 5,
      #'axes.titlesize': font_size,
      #'axes.labelsize': font_size,
      #'figure.titlesize': font_size, # fontsize of the figure title
      'xtick.labelsize': "small", # fontsize of the tick labels
      'ytick.labelsize': "small", # fontsize of the tick labels
      'legend.fontsize': "small"})

def wi1():
        #Interface configuration
    style = {'description_width': 'initial'}
    w10 = wi.IntSlider(min = 10, max = 200, step = 1, value = 30, description='$RR$:')        
    w11 = wi.IntSlider(min = 1, max = 20, step = 1, value = 3, 
                       description='$MFR$ [imp/s]:')
    w12 = wi.IntSlider(min = 10, max = 55, step = 1, value = 35, description='$PFR_1$ [imp/s]:')
    w13 = wi.IntSlider(min = -30, max = 30, step = 1, value = 20, 
                       description='$PFRD$ [imp/s]:')
    w14 = wi.FloatSlider(min = 0.05, max = 0.991, step = 0.01, value = 0.92, 
                         description='$\\tilde{e}_{LR}$ [%]:')
    w15 = wi.IntSlider(min = 1, value = 101, max = 400,step = 1, 
                       description = '# Type I MNs')
    w16 = wi.IntSlider(min = 1, value = 17, description = '# Type IIa MNs')
    w17 = wi.IntSlider(value = 2, description = '# Type IIb MNs')
    w18 = wi.FloatSlider(value = 2, min = 1, max = 10, step = 0.1, 
                         description = '$g_{1}$')
    w19 = wi.Checkbox(value = False, description = 'Same gain for all MNs?')
    ws1 = {'rr':w10, 'mfr':w11, 'firstPFR':w12, 'PFRD':w13, 'RRC':w14,'t1':w15, 't2a':w16, 't2b':w17, 
           'gain_factor':w18, 'gain_CTE':w19}
    ui1 = wi.HBox([wi.VBox([w15, w16, w17, w10, w13, w11, w12, w19, w18, w14])])
    for i in ui1.children[0].children:
        i.layout = wi.Layout(width= '260px')
        i.style = style
        i.continuous_update = False
    return ui1,ws1


def wi2():
    style = {'description_width': 'initial'}
    #Interface configuration
    w42 = wi.FloatSlider(value = 5, min = 0, max = 100, step = 0.1, 
                         description='Plateau/Peak intensity [%]:')
    w43 = wi.IntSlider(value = 0.1, min = 0, max = 3000, step = 100, description='Onset [ms]:')
    w44 = wi.IntSlider(value = 0.1, min = 0, max = 6000, step = 100, description='Plateau on [ms]:')
    w45 = wi.IntSlider(value = 6000, min = 0, max = 6000, step = 100, description='Plateau off [ms]:')
    w46 = wi.IntSlider(value = 6000, min = 0, max = 6000, step = 100, 
                       description='Offset [ms]:', style = style)
    w47 = wi.FloatSlider(value = 0.5, min = 0, max = 2, step = 0.1, description='Frequency [Hz]:')
    w48 = wi.IntSlider(value = 6000, min = 1000, max = 40000, step = 500, 
                       description='Duration [ms]:', style = style)
    w49 = wi.IntSlider(value = 20000, min = 15000, max = 40000, step = 1000, description='Sampling [Hz]:')
    w40 = wi.RadioButtons(options = ['Trapezoidal', 'Sinusoidal'], description = 'Modulation:', style = style)
    ui = wi.VBox([w49,w48,w42,w40,w43,w44,w45,w46,w47])
    ws = {"intensity": w42, "t0": w43, "t1": w44, "t2": w45, "t3": w46, "freq_sin": w47,
          "sample_time": w49, "sim_time": w48, "mode": w40}
    l11 = wi.link((w48, 'value'), (w44, 'max'))
    l12 = wi.link((w48, 'value'), (w45, 'max'))
    l12 = wi.link((w48, 'value'), (w46, 'max'))
    l12 = wi.link((w48, 'value'), (w43, 'max'))
    
    ui.children[0].layout = wi.Layout(width= '350px')
    for i in ui.children:
        i.layout = wi.Layout(width = '320px')
        i.style = style
        i.continuous_update = False
    return ws, ui

def wi3():
    style = {'description_width': 'initial'}
    w0= wi.FloatSlider(value = 30, min = 1, max = 50, step = 0.1, 
                       style = style, description='ISI $cv$ [%]:', 
                       layout = wi.Layout(width= '300px'))
    #w1= wi.FloatSlider(value = 0.02, min = 0.02, max = 0.4, step = 0.01, 
    #                   description='CoV final value:', layout = wi.Layout(width= '500px'), style = style)
    #w2= wi.FloatSlider(value = 1.5, min = 0, max = 10, step = 0.1, 
    #                   description='CoV exp. factor (use zero for cte):', 
    #                   layout = wi.Layout(width= '500px'), style = style)
    w3= wi.FloatSlider(value = 10, min = 0, max = 30, step = 1, 
                       description='Synch. level [%]:', 
                       layout = wi.Layout(width= '300px'), style = style)
    w4= wi.FloatSlider(value = 1.7, min = 0, max = 5,
                       description='Synch $\sigma$ [ms]:', 
                       layout = wi.Layout(width= '300px'), style = style)
    return [w0,w3,w4]

def wi4():
    style = {'description_width': 'initial'}
    w60 = wi.FloatSlider(value = 150, min = 100, max = 1000, step = 50 , style = style,
                         description =  'Muscle CSA [mm$^2$]')
    w61 = wi.FloatSlider(value = 0.4, min = 0.1, max = 0.80, step = 0.01, description = 'Proportion')
    w62 = wi.FloatSlider(value = 0.9, min = 0.05, max = np.pi / 2, step = 0.05, description = 'Theta [rad]')
    w63 = wi.FloatSlider(value = 0.1, min=0, max = 3, step = 0.001, style = style,
                         description = 'Skin layer [mm]')
    w64 = wi.FloatSlider( value = 0.2, min = 0, max = 5, step = 0.001, style = style, 
                         description = 'Fat layer [mm]')
    w66 = wi.RadioButtons(options = ['Circle', 'Ring','Pizza','Ellipse'], value = 'Ring',
                          description = 'CSA morphology:', style = style)
    ui = wi.VBox([w66, w60, w61, w62, w63, w64])
    ws = {"CSA": w60, "prop": w61, "the": w62, "sk": w63, "fa": w64, "morpho": w66}
    for i in ui.children:
        i.layout = wi.Layout(width = '300px')
        i.style = style
        i.continuous_update = False
    ui.layout = wi.Layout(width = '310px')
    return ws,ui

def wi5():
    style = {'description_width': 'initial'}
    w71 = wi.IntSlider(value = 84, min = 10, max = 200, step = 1, description = 'Innervation number ratio',
                       layout = wi.Layout(width = '300px'), style = style, continuous_update = False)
    w72 = wi.FloatSlider(value = 0.7, min = 0, max = 1, step = 0.01, description = 'Type I MU $\mu$',
                         layout = wi.Layout(width = '300px'),style = style, continuous_update = False)
    w73 = wi.FloatSlider(value = 0.5, min = 0.25, max = 1, step = 0.01, description = 'Type I MU $\sigma$',
                         layout = wi.Layout(width = '300px'),style = style, continuous_update = False)
    w74 = wi.FloatSlider(value = 1, min = 0, max = 1, step = 0.01, description = 'Type II MU $\mu$',
                         layout = wi.Layout(width= '300px'), style = style, continuous_update = False)
    w75 = wi.FloatSlider(value = 0.25, min = 0.25, max = 1, step = 0.01, description = 'Type II MU $\sigma$',
                         layout = wi.Layout(width = '300px'), style = style, continuous_update = False)
    vb70 = wi.VBox([w72, w73, w74, w75,w71])
    l7= {'ratio': w71, 't1m': w72, 't1dp': w73, 't2m' : w74, 't2dp' : w75}
    return vb70, l7

def wi6():
    style = {'description_width': 'initial'}
    wi800 = wi.RadioButtons(options = ['1st order', '2nd order'], description = 'Hermite-Rodriguez function:',
                          style = style)
    w80 = wi.FloatSlider(value = 1, min = 0.005, max = 2, step = 0.01, description= 'First MUAP amplitude [mV]:')
    w81 = wi.FloatSlider(value = 11.4, min = 2, max = 150, step = 1, description= 'Last MUAP amplitude [mV]::')
    w82 = wi.FloatSlider(value = 3, min = 0.9, max = 5, step = 0.1, description = 'First MUAP duration [ms]:')
    w83 = wi.FloatSlider(value = 1.4, min = 0.1, max = 5, step = 0.1, description = 'Last MUAP duration [ms]:')
    vb800 = wi.VBox([wi800, w80, w81,w82, w83])
    l8= {'add_hr': wi800, 'v1': w80,'v2': w81, 'd1': w82, 'd2': w83}
    vb800.layout = wi.Layout(width = '360px')
    for i in vb800.children:
        i.layout = wi.Layout(width= '350px')
        i.style = style
        i.continuous_update = False
    return vb800,l8

def wi7():
    style = {'description_width': 'initial'}
    w90 = wi.FloatSlider(value = 5e-3, min = 1e-3, max = 1e-2, step = 1e-3, 
                     readout_format = '.4f',style = style,
                     description = 'Amplitude attenuation factor [mm$^{-1}$]:')
    w91 = wi.IntSlider(value = 100, min = 0, max = 1e3, step = 100, 
                       description = 'Widening factor [mm$^{-1}$]:',style = style)
    ws7 =  {'ampk': w90,'durak': w91}
    ui9 = wi.VBox([w90, w91])
    for i in ui9.children:
        i.layout = wi.Layout(width = '400px')
        i.style = style
        i.continuous_update = False
    return ui9, ws7

def wi8():
    style = {'description_width': 'initial'}
    wi1= wi.Checkbox(value = True, description = 'Add Noise', 
                     layout = wi.Layout(width = '400px'), style = style)
    wi2= wi.FloatSlider(value = 0.05, min = 0, max = 1,step = 0.01, 
                        layout = wi.Layout(width = '400px'), style = style,
                        description = 'Noise standard deviation [mV]')
    wi5= wi.Checkbox(value = True, description = 'Add Filter', 
                     layout = wi.Layout(width = '400px'), style = style)
    wi3= wi.FloatSlider(value = 10, min = 5, max = 50,  
                        layout=wi.Layout(width = '400px'),style = style,
                        description = 'High-pass cutoff frequency [Hz]')
    wi4= wi.FloatSlider(value = 500, min = 500, max = 2000,  
                        layout = wi.Layout(width = '400px'),style = style,
                        description = 'Low-pass cutoff frequency [Hz]')
    return [wi1,wi2,wi3,wi4,wi5]

def wi9(muscle_emg):
    style = {'description_width': 'initial'}
    wi110 = wi.Checkbox(value = True, description = 'Plot moving RMS', 
                        layout = wi.Layout(width= '400px'), 
                        continuous_update = False, style = style)
    wi111 = wi.IntSlider(value = 100, min = 1, max = 500, layout = wi.Layout(width = '400px'),
                         style = style, continuous_update = False, 
                         description = 'Moving RMS window length [ms]:')
    wi112 = wi.Checkbox(value = True, description = 'Plot Spectrogram', continuous_update = False,
                        layout = wi.Layout(width = '400px'),style = style)
    wi113 = wi.Dropdown(options = ['boxcar', 'hamming', 'hann'], value = 'hann', 
                        layout = wi.Layout(width = '400px'), style = style,
                        continuous_update = False,  description = 'Window type:')
    wi114 = wi.IntSlider(value = 120, min = 2, max = 3000, layout = wi.Layout(width = '400px'),style = style,
                        continuous_update = False, description = 'Window length [ms]:')
    wi115 = wi.IntSlider(value = 60, min = 1, max = 1500, layout = wi.Layout(width = '400px'),style = style,
                        continuous_update = False, description = 'Window overlap [ms]:')
    wi116 = wi.Checkbox(value = False, description = 'Plot Welch \' s periodogram',
                       continuous_update = False, layout = wi.Layout(width = '400px'),style = style)
    wi117 = wi.Dropdown(options = ['boxcar', 'hamming', 'hann'], value = 'hann', layout = wi.Layout(width = '400px'),
                       continuous_update = False, style = style, description = 'Window type:')
    wi118 = wi.IntSlider(value = 120, min = 2, max = 3000, layout = wi.Layout(width = '400px'),style = style,
                        continuous_update = False, description = 'Window length [ms]:')
    wi119 = wi.IntSlider(value = 60, min = 1, max = 1500, layout = wi.Layout(width = '400px'),style = style,
                        continuous_update = False, description = 'Window overlap [ms]:')
    end = muscle_emg.t[-1]
    wi1110 = wi.FloatRangeSlider(value = [0, end], min = 0,
                                 max = end,step = 50, 
                                 layout = wi.Layout(width = '500px'),
                                 continuous_update = False, style = style, 
                                 description = 'Analysis interval [ms]')
    wi1111 = wi.Checkbox(value = False, description = 'Plot MU contribution', layout = wi.Layout(width = '400px'),
                        continuous_update = False, style = style)
    wi1112 = wi.BoundedIntText(value = 1, min = 1, max = muscle_emg.LR, step = 1, description = 'MU index #:',
                              continuous_update = False, style = style, layout = wi.Layout(width = '400px'))
    ws11 = {'add_rms': wi110, 'rms_length': wi111, 'add_spec': wi112, 'spec_w': wi113,
        'spec_w_size': wi114, 'spec_ol': wi115, 'add_welch': wi116, 'welch_w': wi117,
        'welch_w_size': wi118, 'welch_ol': wi119, 'a_interval': wi1110, 'add_mu_cont': wi1111, 'mu_c_index': wi1112}
    mu_cont_acc = wi.VBox([wi1111, wi1112])
    moving_average_acc = wi.VBox([wi110, wi111])
    spectrogram_acc = wi.VBox([wi112, wi113, wi114, wi115])
    welch_acc = wi.VBox([wi116,wi117,wi118,wi119])
    acc11 = wi.Tab(children = [moving_average_acc, spectrogram_acc, welch_acc, mu_cont_acc])
    acc11.set_title(0, 'Moving RMS')
    acc11.set_title(1, 'Spectrogram')
    acc11.set_title(2, 'Welch\'s periodogram')
    acc11.set_title(3, 'MUAP train')
    ui9 = wi.VBox([wi1110, acc11])
    l11 = wi.link((ui9.children[1].children[1].children[2], 'value'),
                  (ui9.children[1].children[1].children[3], 'max'))
    l12 = wi.link((ui9.children[1].children[1].children[3], 'value'),
                  (ui9.children[1].children[1].children[2], 'min'))
    l21 = wi.link((ui9.children[1].children[2].children[2], 'value'),
                  (ui9.children[1].children[2].children[3], 'max'))
    l22 = wi.link((ui9.children[1].children[2].children[3], 'value'),
                  (ui9.children[1].children[2].children[2], 'min'))
    return ui9, ws11

def wi10():
    style = {'description_width': 'initial'}
    w111 = wi.IntSlider(value = 130, min = 2, max = 200, step = 1, 
                        description = '$RP$:',
                        continuous_update = False)
    w112 = wi.IntSlider(value = 3, min = 1, max = 5, step = 1, description = '$RT$:',
                        continuous_update = False)
    w113 = wi.IntSlider(value = 90, min = 10, max = 200, step = 1, continuous_update = False,
                          description = '$T_L$ [ms]:')
    w114 = wi.IntSlider(value = 3, min = 1, max = 50, step = 1, description = '$P_1$ [mN]:',
                        continuous_update = False)
    wi115 = wi.RadioButtons(options = ['Exponential', 'Random uniform'], 
                            description = 'Twitch time-to-peak:',
                          style = style)
    vb111 = wi.VBox([w114, w111,wi115,w113, w112])
    l11 = {'RP': w111, 'RT':w112, 'Tl': w113, 'firstP': w114, 'dur_type': wi115}

    for i in vb111.children:
        i.layout = wi.Layout(width = '300px')
        i.style = style
        i.continuous_update = False
    return vb111,l11

def wi11(muscle_force):
    style = {'description_width': 'initial'}
    wit1 = wi.IntSlider(value = 50, min = 5, max = 100, 
                        description = 'First motor unit saturation frequency [imp/s]',
                        style = style, continuous_update = False, 
                        layout = wi.Layout(width = '400px'))
    wit2 = wi.IntSlider(value = 100, min = 20, max =250, 
                        description = 'Last motor unit saturation frequency [imp/s]',
                        style = style, continuous_update = False, 
                        layout = wi.Layout(width = '400px'))
    wit3 = wi.IntSlider(value = 119, max = muscle_force.n-1, min = 0, 
                        description = 'Demonstration motor unit index',
                        style = style, continuous_update = False, 
                        layout = wi.Layout(width = '400px'))
    #wit4 = wi.RadioButtons(options = ['Linear curve','Exponential','1o step resp.','Sigmoidal'],
    #                       description = 'MU force saturation interpolation:',
    #                       style = style, continuous_update = False, 
    #                       layout = wi.Layout(width = '600px'))
    #ui = wi.VBox([wit4,wit1, wit2, wit3])
    ui = wi.VBox([wit1, wit2, wit3])
    #ws = {'fsatf': wit1, 'lsatf': wit2, 'i': wit3, 'sat_interp': wit4}
    ws = {'fsatf': wit1, 'lsatf': wit2, 'i': wit3}
    return ui,ws  

def wi12(muscle_force):
    style = {'description_width': 'initial'}
    wi120 = wi.Checkbox(value = True, description = 'Plot standard deviation', 
                        layout = wi.Layout(width = '400px'), style = style)
    wi122 = wi.Checkbox(value = True, description = 'Plot spectrogram', 
                        layout = wi.Layout(width = '400px'), style = style)
    wi123 = wi.Dropdown(options = ['boxcar', 'hamming', 'hann'], value = 'hann', 
                        layout = wi.Layout(width = '400px'),
                        style = style, description = 'Window type:')
    wi124 = wi.IntSlider(value = 2000, min = 2, max = 5000, step = 5, 
                         layout = wi.Layout(width = '400px'),style = style,
                         description = 'Window length [ms]:')
    wi125 = wi.IntSlider(value = 60, min = 1, max = 2500, step = 5, 
                         layout = wi.Layout(width = '400px'),style = style,
                         description = 'Window overlap [ms]:')
    wi126 = wi.Checkbox(value = False, description = 'Plot Welch \' s periodogram',
                        layout = wi.Layout(width = '400px'), style = style)
    wi127 = wi.Dropdown(options = ['boxcar', 'hamming', 'hann'], value = 'hann', 
                        layout = wi.Layout(width = '400px'),
                        style = style, description = 'Window type:')
    wi128 = wi.IntSlider(value =2000, min = 2, max = 5000, step = 5, 
                         layout = wi.Layout(width = '400px'), style = style,
                         description = 'Window length [ms]:')
    wi129 = wi.IntSlider(value = 60, min = 0, max = 2500, step = 5, 
                         layout = wi.Layout(width = '400px'), style = style,
                         description = 'Window overlap [ms]:')
    wi1210 = wi.FloatRangeSlider(value = [0,muscle_force.t[-1]], min = 0, 
                                 max = muscle_force.t[-1], step = 50, 
                                 layout = wi.Layout(width = '600px'),
                                 style = style, description = 'Analysis interval [ms]')
    wi1211 = wi.Checkbox(value = False, description = 'Plot MU force', 
                         layout = wi.Layout(width = '400px'),style = style)
    wi1212 = wi.BoundedIntText(value = 1, min = 1, max = muscle_force.LR, step = 1, 
                               description = 'Motor unit index #:',
                               style= style, layout=wi.Layout(width= '400px'))
    ws = {'add_rms': wi120, 'add_spec': wi122, 'spec_w': wi123, 'spec_w_size': wi124, 
          'spec_ol': wi125, 'add_welch': wi126, 'welch_w': wi127,
          'welch_w_size': wi128, 'welch_ol': wi129, 'a_interval': wi1210,
          'add_mu_c': wi1211, 'mu_index': wi1212}
    moving_average_acc2 = wi.VBox([wi120])
    spectrogram_acc2 = wi.VBox([wi122, wi123, wi124, wi125])
    welch_acc2 = wi.VBox([wi126, wi127, wi128, wi129])
    mu_c2 = wi.VBox([wi1211, wi1212])
    acc12 = wi.Tab(children = [moving_average_acc2, spectrogram_acc2, welch_acc2, mu_c2])
    acc12.set_title(0, 'Standard deviation')
    acc12.set_title(1, 'Spectrogram')
    acc12.set_title(2, 'Welch\'s periodogram')
    acc12.set_title(3, 'Motor unit force')
    ui12 = wi.VBox([wi1210, acc12])
    l211 = wi.link((ui12.children[1].children[1].children[2], 'value'),
                   (ui12.children[1].children[1].children[3], 'max'))
    l212 = wi.link((ui12.children[1].children[1].children[3], 'value'),
                   (ui12.children[1].children[1].children[2], 'min'))
    l221 = wi.link((ui12.children[1].children[2].children[2], 'value'), 
                   (ui12.children[1].children[2].children[3], 'max'))
    l222 = wi.link((ui12.children[1].children[2].children[3], 'value'), 
                   (ui12.children[1].children[2].children[2], 'min'))
    return ui12, ws

def wi13():
    w0 = wi.Checkbox( value = True, description = 'Save simulation config.')
    w1 = wi.Checkbox( value = True, description = 'Save spike times')
    w2 = wi.Checkbox( value = True, description = 'Save sEMG')
    w3 = wi.Checkbox( value = True, description = 'Save muscle force')
    w4 = wi.Text(value = time.strftime("%d_%m_%Y-%H_%M_%S"), placeholder = 'Type something',
                   description = 'Folder name:', disabled = False)
    return [w0,w1,w2,w3,w4]



def MedFreq(self,freq,psd):
        cum = cumtrapz(psd,freq,initial = 0)
        f = np.interp(cum[-1]/2,cum,freq)
        mfpsdvalue = cum[-1]/2
        return f,mfpsdvalue

class sim_results(object):
    def __init__(self,mnpool,muscle_emg,muscle_force):
        self.mnpool = mnpool
        self.muscle_emg = muscle_emg
        self.muscle_force = muscle_force
        self.config = {}
        
    def update_config(self):
        self.config.update(self.mnpool.save_config())
        self.config.update(self.muscle_emg.save_config())
        self.config.update(self.muscle_force.save_config())
        
    def save_results(self,save_conf, save_spikes, save_emg, save_force, folder_name):
        self.update_config()
        sim_res = "simulation_results/"
        results_dir = sim_res  + folder_name + '/'
        if not os.path.isdir(sim_res):
            os.mkdir(sim_res)
        if os.path.isdir(results_dir):
            print('Folder name {} exists. Try another folder name.'.format(folder_name))
        else:
            os.mkdir(results_dir)
            results_dir = results_dir + folder_name + '-config_reference/'
            os.mkdir(results_dir)
            print('{} Folder created.'.format(folder_name))
            np.savetxt(results_dir + 't.txt', self.mnpool.t, fmt = '%.2e')
            if save_conf == True:
                try:
                    df = pd.Series(self.config, name ='Configuration')
                    df = pd.DataFrame(df)
                    df.to_csv(results_dir + 'Configuration.csv')
                    print('Configuration.csv created.')
                except:
                    print("Can't save configuration. Before saving, you need to generate simulation output.") 
            if save_spikes == True:
                if self.mnpool.neural_input == []:
                    print("Can't save spike times.")
                    print("Please click on 'Run interact' in the section 'Motor unit spike trains'.")
                else:
                    i = 1
                    with open(results_dir + 'spike_times.txt', 'w') as f:
                        for spikes in self.mnpool.neural_input:
                            if not spikes == []:
                                for pa in spikes:
                                    f.write("{} {:.6f}\n".format(i, pa))
                                i += 1
                    print('spike_times.txt file created and saved.')
            if save_emg == True:
                if self.muscle_emg.emg == []:
                    print("Can't save surface EMG.")
                    print("Please click on 'Run interact' in the section 'Generation of sEMG'.")
                else:
                    np.savetxt(results_dir + 'raw_emg.txt', self.muscle_emg.emg, fmt='%.6e')
                    print('Surface emg file raw_emg.txt saved.')
            if save_force == True:
                if self.muscle_force.muscle_force == []:
                    print("Can't save muscle force.")
                    print("Please click on 'Run interact' in the section 'Generation of muscle force'.")
                else:
                    np.savetxt(results_dir + 'muscle_force.txt', 
                               self.muscle_force.muscle_force, fmt='%.6e')
                    print('Muscle force file muscle_force.txt saved.')
            