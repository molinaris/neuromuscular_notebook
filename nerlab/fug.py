import numpy as np
from math import log
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Phenonomenologycal approach to model recruitment and rate coding organization of a motorneuron pool 
class Phemo(object):
    
    def __init__(self):
        self.t1 = 101 # Number of type I motor units
        self.t2a = 17 # Number of type IIa motor units
        self.t2b = 2 # Number of type IIb motor units
        self.n = self.t1 + self.t2a + self.t2b # Number of motor units in the pool
        self.sampling = 2e4 # Sampling Frequency [Hz]
        self.dt = 1/2e4 # simulation step time [s]
        self.t = np.arange(0,5e3,self.dt*1e3) # time array
        self.rr = 30 #range of recruitment
        self.pfrd = 20 #peak firing rate difference [Hz]
        self.mfr = 3 #Minimum firing rate [Hz]
        self.firstPFR = 35 # First recruited Peak firing rate [Hz]
        self.gain_cte = False # If true, all motor unit gain (exicatory x firing rate) are equal 
        self.gain_factor= 2 # Gain factor
        self.LR = 1 # Last recruited
        self.rrc = 0.67 #Recruitment range condition [%]
        self.ISI_limit = 15 # Minimum inter spike interval  [ms]
        self.recruitThreshold() #Defines Recruitment threshold for all motor units
        self.peakFireRate() #Defines peak firing rate for all motor units
        self.recruitmentRangeCondition() # Defines Maximum excitatory drive and gain for all motor units
        self.neural_input = [] #Spike train for all motor units
        self.intensity = 0 # excitatory drive intensity
        self.config = {} # Configuration 
        self.CV = 0

    # FUNCTION NAME: recruitThreshold
    # FUNCTION DESCRIPTION: Defines the recruitment threshold for each motorneuron of the pool.
    #                       Equal to the original proposed by Fuglevand et Al, 1993.
    def recruitThreshold(self): 
        a=np.log(self.rr)/self.n
        self.rte = np.zeros(self.n) # Recruitment threshold excitation
        for i in range(self.n):
            self.rte[i]=np.exp(a*(i+1))
        
    # FUNCTION NAME: peakFireRate
    # FUNCTION DESCRIPTION: Defines the Peak firing rate for every motorneuron in the pool.
    #                       Equal to the original proposed by Fuglevand et Al, 1993.
    def peakFireRate(self):
        self.pfr = np.zeros(self.n)
        for i in range(self.n):
            self.pfr[i] = self.firstPFR - self.pfrd*self.rte[i]/self.rte[-1]
        
    # FUNCTION NAME: recruitmentRangeCondition
    # FUNCTION DESCRIPTION: Defines the Firing rate x excitatory drive gain for each motorneuron
    #                       and the maximum exitatory drive
    def recruitmentRangeCondition(self):
        self.Emax = self.rte[-1]/self.rrc
        var_gain = np.linspace(self.gain_factor,1,len(self.pfr))
        self.gain = var_gain*(self.pfr - self.mfr)/(self.Emax-self.rte)
        if self.gain_cte:
            last_gain = self.gain[-1]
            for i in range(self.n):
                self.gain[i] = last_gain
                
                    
    # FUNCTION NAME: fireRate
    # FUNCTION DESCRIPTION: Calculates the firing rate of the motorneuron pool over time
    # INPUT PARAMS:  1) E: Excitatory drive over simulation time (a.u.)[np array]
    # OUTPUT PARAMS: 1) fr: Firing rate for each motorneuron over time as function of E
    #                   (Hz)(2d numpy array)
    def fireRate(self,E):
        t_size = len(E)
        lastrec = np.zeros(t_size)
        for i in range(len(E)):
            for j in range(len(self.rte)):
                if E[i] >= self.rte[j]:
                    lastrec[i] = j + 1
        self.LR = int(max(lastrec))
        fr = np.zeros((self.LR,t_size))
        for i in range(self.n):
            for j in range(t_size):
                if (E[j] > self.rte[i]):
                    fr[i][j] = self.gain[i]*(E[j]-self.rte[i]) + self.mfr
                    if (self.pfr[i] < fr[i][j]):
                        fr[i][j] = self.pfr[i]
        return fr

    def graph_FRxExcitation(self):
        """ FUNCTION NAME: FRxExcitation
            FUNCTION DESCRIPTION: plot recruitment and firing rate organization of the motorneuron pool
        """
        e = np.linspace(0,self.Emax,200)
        fr = self.fireRate(e)
        
        plt.figure(figsize = (5,4))
        for i in range(self.t1-1):
            plt.plot(100*e/self.Emax, fr[i], c = '#4C72B0')
        plt.plot(100*e/self.Emax, fr[self.t1-1], c = '#4C72B0', label = "MN I")   
        for i in range(self.t1, self.t1 + self.t2a-1):
            plt.plot(100*e/self.Emax, fr[i], c = '#55A868', ls = "--")
        plt.plot(100*e/self.Emax, fr[self.t1 + self.t2a - 1], c = '#55A868', ls = "--", label = "MN IIa")  
        for i in range(self.t1 + self.t2a, self.n - 1):
            plt.plot(100*e/self.Emax, fr[i], c = '#C44E52', ls = "-.")
        plt.plot(100*e/self.Emax, fr[self.n - 1], c = '#C44E52', ls = "-.", label = "MN IIb")  
        plt.xlabel('Excitatory drive [%]')
        plt.ylim(self.mfr)  
        plt.xlim(0,100)
        plt.ylabel('Firing Rate [imp/s]')
        plt.legend(loc=2)
        
    # FUNCTION NAME: view_organization
    # FUNCTION DESCRIPTION: update and plot recruitment and firing rate
    #                       organization of the motorneuron pool
    # INPUT PARAMS:  1) rr: Range of recruitment excitations of the motorneuron pool(x fold)[int]
    #                2) mfr: Minimal firing rate for every motorneuron of the pool (Hz)[int]
    #                3) firstPFR: Peak firing rate of the first recruited motorneuron (Hz)[int]
    #                4) PFRD: Desired peak firing rate difference between first and last 
    #                   motorneuron (Hz)[int]
    #                5) RRC: Defines the relative excitatory drive necessary to recruit the last
    #                   motorneuron (%) [float]
    #                6) t1: quantity of type I motorneurons in the pool [int]
    #                7) t2a: quantity of type IIa motorneurons in the pool [int]
    #                8) t2b: quantity of type IIb motorneurons in the pool [int]
    #                9) gain_factor: gain factor for the first recruited motorneuron [float]
    #                10) gain_CTE: Flag to define all motorneuron gains equal to the last 
    #                   recruited (flag) [boolean]
    def view_organization(self,rr, mfr, firstPFR, PFRD, RRC, t1, t2a, t2b, gain_factor,gain_CTE):
        self.t1 = t1
        self.t2a = t2a
        self.t2b = t2b
        self.n = t1 + t2a + t2b
        self.rr = rr
        self.pfrd = PFRD
        self.mfr = mfr
        self.firstPFR = firstPFR
        self.gain_cte = gain_CTE
        self.gain_factor= gain_factor
        self.rrc = RRC
        self.recruitThreshold()
        self.peakFireRate()
        self.recruitmentRangeCondition()
        self.graph_FRxExcitation()
 
    # FUNCTION NAME: excitation
    # FUNCTION DESCRIPTION: caclulates the excitatory drive over the simulation time
    # INPUT PARAMS:  1) t0: excitatory drive onset time (ms)[float]
    #                2) t1: excitatory drive plateau on time (ms)[float]
    #                3) t2: excitatory drive plateau off time (ms)[float]
    #                4) t3: excitatory drive offset time (ms)[float]
    #                5) f: sinusoidal frequency excitatory drive time variation (Hz)[float]
    #                6) mode: Excitatory drive mode, 'trap' for trapezoidal or 'sin' for
    #                   sinusoidal excitation curve [string]
    #                7) intensity: plateou relative excitatory drive for 'trap' mode and
    #                   peak excitatory drive for 'sin' mode (%) [float]
    def excitation_curve(self,t0,t1,t2,t3,f,mode,intensity):
        dt =  self.dt
        Emax = self.Emax
        self.intensity = intensity
        self.t00 = t0
        self.t01 = t1
        self.t02 = t2
        self.t03 = t3
        self.mode = mode
        self.e_freq = f
        ramp_init = int(t0/dt)
        plateau_init = int(t1/dt)
        dramp_init = int(t2/dt)
        dramp_end = int(t3/dt)
        ramp = (t1-t0)/dt
        if ramp == 0:
            stepup = 0
        else:
            stepup = intensity*Emax/ramp
        dramp = (t3-t2)/dt
        if dramp == 0:
            stepdown = 0
        else:
            stepdown = intensity*Emax/dramp
        if (mode == "Trapezoidal"):
            for i in range(0,ramp_init):
                self.E[i] = 0
            for i in range(ramp_init,plateau_init):
                self.E[i] = stepup*(i-ramp_init)
            for i in range(plateau_init,dramp_init):
                self.E[i] = intensity*Emax
            for i in range(dramp_init,dramp_end):
                self.E[i] = intensity*Emax - stepdown*(i+1-dramp_init)
            for i in range(dramp_end,self.t_size):
                self.E[i] = 0
        if (mode == "Sinusoidal"):
            for i in range(self.t_size):
                self.E[i] = intensity*Emax/2 + intensity*Emax/2*np.sin(2*np.pi*f*self.t[i])
       
    def view_excitatory(self,intensity, t0, t1, t2, t3, freq_sin, sample_time, sim_time,mode):
        """    # FUNCTION NAME: view_excitatory
        # FUNCTION DESCRIPTION: caculates and plot the excitatory drive over the simulation time
        # INPUT PARAMS:  1) intensity: plateou relative excitatory drive for 'trap' mode and
        #                   peak excitatory drive for 'sin' mode (%) [float]
        #                2) t0: excitatory drive onset time (ms)[float]
        #                3) t1: excitatory drive plateau on time (ms)[float]
        #                4) t2: excitatory drive plateau off time (ms)[float]
        #                5) t3: excitatory drive offset time (ms)[float]
        #                6) freq_sin: sinusoidal frequency excitatory drive time variation (Hz)[float]
        #                7) sample_time: simulation sampling time [Hz]
        #                8) sim_time: total simulation time [float]
        #                6) mode: Excitatory drive mode, 'Trapezoidal' or 'Sinusoidal' 
        #                   excitation curve [string]  
        """
        self.sampling = sample_time
        self.sim_time = sim_time
        self.dt = 1e3/sample_time
        self.t = np.arange(0, self.sim_time, self.dt) #Time Array in [ms]
        self.t_size = len(self.t)
        self.E = np.zeros(self.t_size)
        self.excitation_curve(t0,t1,t2,t3,freq_sin*1e-3,mode,intensity/100)
        plt.figure(figsize=(4,4))
        plt.plot(self.t, self.E/self.Emax * 100)
        plt.xlabel('Time [ms]')
        plt.ylabel('Excitation [%]')      

    # FUNCTION NAME: neuralInput
    # FUNCTION DESCRIPTION: Calculates the motorneuron pool discharge times
    #                       over the simulation time.
    def neuralInput(self):
        x,y = self.fr.shape
        self.neural_input = []
        for i in range (x): # for each MU 
            spike_train = []
            next_spike = 0
            flag = 0
            for j in range(y): # for each instant
                if (self.fr[i][j]>0 and self.t[j]>next_spike):
                    sigma = self.CV*1e3/self.fr[i][j]
                    if not spike_train:
                        next_spike = self.t[j] + self.add_spike(sigma,self.fr[i][j])
                        spike_train.append(next_spike)
                        k = 0
                    else:
                        if (flag == 1 and (self.t[j] - spike_train[k] > 2)):
                            flag = 0
                            next_spike = self.t[j] + self.add_spike(sigma,self.fr[i][j])
                        else:
                            next_spike = self.add_spike(sigma,self.fr[i][j]) + spike_train[k]
                        if (next_spike > self.t[-1]):
                            break
                        if (next_spike -  spike_train[k] > 2):
                            spike_train.append(next_spike)
                            k = k + 1
                if (self.fr[i][j] == 0):
                    flag = 1 #MU stopped
            self.neural_input.append(np.asarray(spike_train))

    # FUNCTION NAME: add_spike
    # FUNCTION DESCRIPTION: Calculates the inter spike interval (ISI) for the next 
    #                       discharge time of a motorneuron
    # INPUT PARAMS:  1) sigma: standard deviation of the ISI (ms) [float]
    #                2) FR: Mean Firing rate of a motorneuron at a given 
    #                   time (Hz) [float]
    # OUTPUT PARAMS: 1) New inter spike interval (ms) [float]
    def add_spike(self,sigma,FR):
        sig = np.random.normal(0,sigma)
        while abs(sig) > sigma*3.9:
            sig = np.random.normal(0,sigma)
        return  sig + 1e3/FR

    # FUNCTION NAME: synchSpikes
    # FUNCTION DESCRIPTION: Apply the synchronization algorithm [yao et Al, 2001]
    #                       to the discharge times of a MN pool. 
    # INPUT PARAMS:  1) synch_level: desired level of synchrony (%) [float]
    #                2) sigma: standard deviation of the normal distribution add to the 
    #                   synchronized discharge (ms) [float]
    def synchSpikes(self,synch_level,sigma):
        pool = np.arange(0,self.LR, dtype=int)  #Create the array of all recruited motor units indexes
        np.random.shuffle(pool) #Shuffle the Pool
        synch_pool_limit = int(synch_level*self.LR)   #set the limit number of synched MUs
        for i in pool:      #for all active MU in the pool do:
            i_index= np.argwhere(pool == i)     #localize the index of [i]
            synchin_pool = np.delete(pool,i_index)    #remove the reference MU from the pool indexes
            #Define the reference spikes to synchronization of the mu[i]
            ref_spikes = np.random.choice(self.neural_input[i], 
                                          int(synch_level*len(self.neural_input[i])), 
                                          replace = False)
            for j in ref_spikes:        #for all the reference spikes of MU[i] do:
                np.random.shuffle(synchin_pool) #shuffle the order of the pool to be synchronized
                synched_MUs = 0 #Synchronized motor units
                w = 0 #synchronizing pool index
                while (synched_MUs < synch_pool_limit and w < self.LR-1):
                    k = synchin_pool[w]
                    w += 1
                    difs = abs(self.neural_input[k] - j) #Vector of differences between discharge ref and candidate mn.
                    minimum = min(difs) #Minimum diference between then
                    min_index= np.argwhere(difs == minimum)[0][0] # Index of the minimum 
                    #If k motorneuron(mn) candidate is recruited and we did not reach the total 
                    # quantity of synched mns,
                    if self.ISI_limit == 0:
                        adjusted_spike_time = j + np.random.normal(0,sigma) # New spike position
                        self.neural_input[k][min_index] = adjusted_spike_time
                        synched_MUs += 1
                    else:
                        if (minimum < self.ISI_limit):   
                            adjusted_spike_time = j + np.random.normal(0,sigma) # New spike position
                            self.neural_input[k][min_index] = adjusted_spike_time
                            synched_MUs += 1
    
    # FUNCTION NAME: view_neural_command
    # FUNCTION DESCRIPTION: Plot neural command and other performance indicators
    # INPUT PARAMS:  1) CoV: Initial cv to be used in the interpolation (if cv_factor = 0, this value
    #                   will be used for all excitatory drives [float]
    #                2) synch_level: desired level of synchrony (%) [float]
    #                3) sigma: standard deviation of the normal distribution add to the 
    #                   synchronized discharge (ms) [float]
    def view_neural_command(self, CoV , synch_level, sigma):
            print("Processing...")
            self.CV = CoV
            self.synch_level = synch_level
            self.synch_sigma = sigma
            self.fireRate(self.E) #Defines the mean firing rate
            self.CV = CoV/100
            self.fr = self.fireRate(self.E)
            self.neuralInput() #Generates the neural input to the muscles
            self.synchSpikes(synch_level/100, sigma) #Promotes synchronism between MU
            #Inter Spike interval Analysis
            ISI = [np.diff(mu_isi) for mu_isi in self.neural_input if mu_isi != []]
            isi_hist = [item for mu_isi in ISI for item in mu_isi]
            isi_mean = [np.mean(mu_isi) for mu_isi in ISI]
            isi_std = [np.std(mu_isi) for mu_isi in ISI]
            isi_cv = [mu_isi_std / mu_isi_mean for mu_isi_std, mu_isi_mean in zip(isi_std, isi_mean)]
            clear_output()
            f, axes = plt.subplots(4, 1, figsize=(8,8))
            axes[0] = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
            axes[1] = plt.subplot2grid((4, 1), (2, 0))
            axes[2] = plt.subplot2grid((4, 1), (3, 0))
            axes[0].eventplot(self.neural_input)
            plt.sca(axes[0])
            plt.ylabel("MU #")
            plt.xlabel('Time (ms)')
            plt.xlim(0, self.t[-1])
            plt.ylim(0, self.LR+1)
            plt.sca(axes[1])
            plt.hist(isi_hist, bins = np.arange(0, 500, 10), edgecolor = 'k')
            plt.annotate("ISI mean: {:.2f}".format(np.mean(isi_mean)), xy=(0.7,0.9), xycoords = ("axes fraction"))
            plt.annotate("ISI Std. Dev.: {:.2f}".format(np.mean(isi_std)), xy=(0.7,0.77), xycoords = ("axes fraction"))
            plt.annotate("ISI Coef. Var.: {:.2f}".format(np.mean(isi_cv)), xy=(0.7,0.64), xycoords = ("axes fraction"))    
            plt.ylabel('Count')
            plt.xlabel('Interspike Interval (ms)')
            plt.sca(axes[2])
            plt.plot(np.asarray(isi_cv)*100, '.',marker = 'o')
            plt.ylabel('ISI CoV [%]')
            plt.xlabel('MN index')
            plt.tight_layout()
            f.subplots_adjust(hspace=0.40)
            f.align_ylabels()



    # FUNCTION NAME: save_config
    # FUNCTION DESCRIPTION: Generate dictionary with motorneuron pool model organization
    # OUTPUT PARAMS:        1) Dictionary with motorneuron pool model parameters
    def save_config(self):
        try:
            self.config.update({'# Type I MU': self.t1,
                     '# Type IIa MU': self.t2a,
                     '# Type IIb MU': self.t2b,
                     'RR': self.rr,
                     'PFRD [Hz]': self.pfrd,
                     'MFR [Hz]': self.mfr,
                     'PFR_1 [Hz]': self.firstPFR,
                     'Same gain for all MNs?':self.gain_cte,
                     'g_1 [a.u.]':self.gain_factor,
                     'e_LR [%]': self.rrc,
                     'Plateau/peak intensity [%]': self.intensity,
                     'Onset [ms]': self.t00,
                     'Plateau on [ms]':self.t01,
                     'Plateau off [ms]':self.t02,
                     'Offset [ms]':self.t03,
                     'Modulation':self.mode,
                     'Frequency [Hz]':self.e_freq,
                     'Sampling [Hz]': self.sampling,
                     'Duration [ms]': self.sim_time,
                     'ISI cv':self.CV,
                     'synch. level [%]': self.synch_level,
                     'Synch. sigma [ms]': self.synch_sigma})


        except:
            print('Couldn\'t save motorneuron pool parameters, try to click \'run interact\' on neural command generation cell.')
        return self.config