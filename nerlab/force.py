import numpy as np
from math import log
from scipy.signal import lfilter, welch, get_window, spectrogram
from scipy.integrate import cumtrapz
from scipy.optimize import newton
import matplotlib.pyplot as plt
from IPython.display import clear_output
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Muscle_force (object):
    def __init__(self, mnpool):
        self.t = mnpool.t
        self.n = mnpool.n
        self.rr = mnpool.rr
        self.dt = mnpool.dt
        self.pfr = mnpool.pfr
        self.sampling = mnpool.sampling
        self.neural_input = mnpool.neural_input
        self.LR = mnpool.LR
        self.mnpool = mnpool
        self.newton_target = 0.999
        self.fP = 3e-3 #[N]
        self.RP = 130
        self.RT = 3
        self.Tl = 90 #[ms]
        self.dur_type = 'Exponential'
        self.forcePeakAmp() #[a.u.]
        self.forceContracTime() #[s]
        self.muscle_force = []
        self.config = {}
        self.T = []
        
    def forcePeakAmp(self):
        """ FUNCTION NAME: forcePeakAmp
            FUNCTION DESCRIPTION: Generates the reference peak force 
            of the twitch for each motor unit in the pool. """
        self.P = np.zeros(self.n)
        b = np.log(self.RP)/self.n
        for i in range(1, self.n + 1):
            self.P[i-1] = np.exp(b * i)
            
    def  forceContracTime(self):
        """ FUNCTION NAME: forceContracTime
            FUNCTION DESCRIPTION: Generates the array for all motor 
            units in the pool with the reference contraction time that 
            takes for the twitch force reach its peak. """
        self.T = np.zeros(self.n)
        if (self.dur_type == 'Exponential'):
            c = log(self.RP,self.RT)
            for i in range(1, self.n + 1):
                self.T[i-1]=self.Tl*(1/self.P[i-1])**(1/c)
        else:
            i = 0
            while i < self.n:
                val = np.random.normal(self.CT_mean,self.CT_std)
                if val <= self.Tl and val >= self.Tl/self.RT:
                    self.T[i] = val
                    i += 1

    def twitchForce(self,t,i):
        """ FUNCTION NAME: twitchForce
            FUNCTION DESCRIPTION: Simluates the motor unit twitch as a
            second order critically damped system as the response for 
            a motor unit discharge.
            INPUT PARAMS:   1) t: Simulation time array
                            2) i: motor unit index [int]
            OUTPUT PARAMS:  1) twitchforce: simulated twitch force over 
                                time."""

        twitchforce = np.zeros(len(t))
        for j in range(len(t)):
            temp            = np.exp(1-t[j]/self.T[i])/self.T[i]
            twitchforce[j]  = self.fP*self.P[i]*t[j]*temp
        return twitchforce
    
    def get_mu_twitch(self,RP, RT, Tl, firstP, dur_type,g_mean,g_std):
        self.fP         = firstP * 1e-3
        self.RP         = RP
        self.RT         = RT
        self.Tl         = Tl
        self.CT_mean    = g_mean
        self.CT_std     = g_std
        self.dur_type   = dur_type
        self.t          = self.mnpool.t
        self.dt         = self.mnpool.dt
        self.forcePeakAmp()
        self.forceContracTime() #[s]

    def view_mu_twitch(self,RP, RT, Tl, firstP, dur_type,g_mean,g_std):
        self.get_mu_twitch(RP, RT, Tl, firstP, dur_type,g_mean,g_std)
        x       = np.arange(0, self.Tl * 4)
        y       = np.arange(self.n)
        z       = np.zeros((y.shape[0],x.shape[0]))
        fig,ax  = plt.subplots(3,1,figsize=(5,9))
        plt.sca(ax[0])
        for i in range(self.n):
            z[i] = self.twitchForce(x, i)
            if i % int(self.n/5) == 0:
                plt.plot(x,z[i]*1e3, label = 'MU${}_{%d}$'%(i+1))
        plt.plot(x,z[i]*1e3, label = 'MU${}_{%d}$'%(i+1))
        plt.xlabel('Time [ms]')
        plt.ylabel('Force [mN]')
        plt.legend(loc=1)
        plt.sca(ax[1])
        plt.plot(y,self.fP*self.P * 1e3)
        plt.xlabel('MU $i$')
        plt.ylabel('Twitch amp. [mN]')
        plt.sca(ax[2])
        plt.plot(self.fP*self.P * 1e3,self.T,'.')
        plt.xlabel('Twitch amplitude [mN]')
        plt.ylabel('Twitch time-to-peak [ms]')
        plt.tight_layout()

    def sat_interpol(self,fsatf,lsatf,sat_interp='Exponential'):
        """ FUNCTION NAME: sat_interpol
            FUNCTION DESCRIPTION: Interpolate with different styles 
            between first and last MU force saturation frequency."""
        self.sat_interp = sat_interp
        self.fsatf      = fsatf
        self.lsatf      = lsatf
        if self.sat_interp == 'Linear curve':
            self.mu_saturation = np.linspace(self.fsatf,self.lsatf,self.n)
        if self.sat_interp == 'Exponential':
            a=np.log(self.rr)/self.n
            rte = np.zeros(self.n)
            self.mu_saturation = np.zeros(self.n)
            for i in range(self.n):
                rte[i]  = np.exp(a*(i+1))
                temp    = rte[i]*(self.lsatf-self.fsatf)/self.rr
                self.mu_saturation[i] = temp + self.fsatf
        if self.sat_interp == '1o step resp.':
            mu_ind = np.linspace(1,self.n,self.n)
            temp   = (self.lsatf-self.fsatf)*(1-np.exp(-mu_ind*5/self.n))
            self.mu_saturation = temp+self.fsatf
        if self.sat_interp == 'Sigmoidal':
            mu_ind = np.linspace(-int(self.n/4),int(3*self.n/4),self.n)
            temp   = (self.lsatf-self.fsatf)/2*self.sig(0.2,mu_ind)
            self.mu_saturation = temp+(self.lsatf+self.fsatf)/2

    def gen_mu_force(self,mu_spikes,P,T):
        """ FUNCTION NAME: gen_mu_force
            FUNCTION DESCRIPTION: This function generates the motor 
            unit force over the simulation time. This function is like
             the original proposed by Cisi and Kohn, 2008.
            INPUT PARAMS:   1) mu_spikes: List with the discharge 
                            times of a motorneuron.
                            2) i : motor unit index [int]
            OUTPUT PARAMS:  1) mu_force: simulated motor unit force 
                            over time."""
        mu_force = np.zeros(len(self.t))
        spikes   = np.zeros(len(self.t))
        for spike_times in mu_spikes:
            index = int(spike_times/self.dt)
            if index >= len(self.t):
                index = len(self.t) -1
            spikes[index] = 1/self.dt
        B = np.array([0,P*self.dt**2/T*np.exp(1-self.dt/T)])
        A = np.array([1, -2*np.exp(-self.dt/T), np.exp(-2*self.dt/T)])
        mu_force = lfilter(B,A,spikes) 
        return mu_force
            
    def find_tetanic_parameters(self,i,c_init):
        """ FUNCTION NAME: find_tetanic_parameters
            FUNCTION DESCRIPTION: Find the parameter c which saturates
            the motor unit force at a given frequency.
            INPUT PARAMS:  1) i: motor unit index
                           4) c_init: c initial guess used in the 
                           secant (newton) method function
            OUTPUT PARAMS: 1) n_c: c value in which the sigmoidal 
                           function will saturate the motor unit force.
                           2) fsat_freq1_max: maximum force generated 
                           by the motor unit after it is passed by the
                           sigmoidal (sig) function."""
        spikes      = np.arange(0,5e3,1e3/self.mu_saturation[i])
        mu_force    = self.gen_mu_force(spikes,1,self.T[i])
        n_c         = newton(   self.newton_f,c_init,args=(mu_force,),
                                tol=1e-5,maxiter=500)
        freqs       = np.arange(0,5e3,1e3/1)
        mu_force_freq1 = self.gen_mu_force(freqs,1,self.T[i])
        fsat_freq1_max = max(self.sig(n_c,mu_force_freq1))
        return n_c,fsat_freq1_max
    
    def sig(self,c,force):
        """ FUNCTION NAME: sig
            FUNCTION DESCRIPTION: Sigmoidal function used to simulate 
            the non-linear relationship between motor unit discharge 
            rate and generated motor unit twitch force;
            INPUT PARAMS:  1) c: parameter used to adjust the the 
            sigmoidal function so the motor unit twitch force saturates 
            at a determined motor unit discharge rate
                           2) force: The simulated motor unit force over
                           time. 
            OUTPUT PARAMS: 1) the function returns the motor unit force 
                           with the non-linear saturation feature. """
        expfc = np.exp(-force*c)
        return (1-expfc)/(1+expfc)
    
    def gen_sat_demo(self,freqs_demo,i,c=0):
        """ output in mN """
        self.t      = self.mnpool.t
        self.dt     = self.mnpool.dt
        n           = len(freqs_demo)
        max_force   = np.zeros(n)
        c_input, tet_twitch_ratio = self.find_tetanic_parameters(i, 0.20)
        for j,freq in enumerate(freqs_demo):
            spikes       = np.arange(0, self.t[-1], 1e3/freq)
            pre_force    = self.gen_mu_force(spikes, 1, self.T[i])
            temp         = self.fP*self.P[i]/tet_twitch_ratio
            mu_force     = self.sig(c_input, pre_force)*temp
            max_force[j] = max(mu_force)*1e3 # [mN]
        return max_force,c_input

    def view_saturation(self,fsatf, lsatf,i):
        self.pfr    = self.mnpool.pfr
        i           = i - 1
        self.sat_interpol(fsatf,lsatf)
        self.t      = self.mnpool.t
        self.dt     = self.mnpool.dt
        f, axes     = plt.subplots(1, 2, figsize=(9, 4))
        plt.sca(axes[0])
        plt.plot(np.arange(1,self.n+1),self.mu_saturation)
        plt.xlabel('MU $i$')
        plt.xlim(1,self.n)
        plt.ylabel('Force saturation frequency [imp/s]')
        freqs_demo = np.linspace(1,self.pfr[i],5)
        c_input, tet_twitch_ratio = self.find_tetanic_parameters(i, 0.20)
        plt.sca(axes[1])
        for freq in freqs_demo:
            spikes      = np.arange(0, self.t[-1], 1e3 / freq)
            pre_force   = self.gen_mu_force(spikes, 1, self.T[i])
            temp        = self.fP * self.P[i] / tet_twitch_ratio
            mu_force    = self.sig(c_input, pre_force) * temp
            plt.plot(self.t, mu_force*1e3, label='{:.0f} Hz'.format(freq))
            if (freq == 1):
                new_twitch_amp  = max(mu_force)
            if (freq == self.pfr[i]):
                pfr_max_force   = max(mu_force)*1e3
        plt.ylabel('Force [mN]')
        plt.xlabel('Time [ms]')
        plt.ylim(top=pfr_max_force+1)
        plt.xlim(0,1e3)
        plt.title("MU${}_{%d}$"%(i+1))
        plt.legend(loc=4)
        plt.tight_layout()
        spikes    = np.arange(0, self.t[-1], 1e3/self.mu_saturation[i])
        pre_force = self.gen_mu_force(spikes, 1, self.T[i])
        temp      = self.fP*self.P[i]/tet_twitch_ratio
        tet_max_force = max(self.sig(c_input, pre_force)*temp)
        print('MU twitch amplitude: {:.2f} mN'.format(new_twitch_amp*1e3))
        print('MU twitch time-to-peak: {:.2f} ms'.format(self.T[i]))
        print('MU tetanic force: {:.2f} mN'.format(tet_max_force*1e3))
        print('MU force at PFR: {:.2f} mN'.format(pfr_max_force))
        tt_ratio = new_twitch_amp*100/tet_max_force
        print('MU twitch/tetanus ratio: {:.2f} %'.format(tt_ratio))
        mu_sat = self.mu_saturation[i]
        print('MU saturation frequency: {:.2f} Hz'.format(mu_sat))
        
    def newton_f(self,c,force):
        """ FUNCTION NAME: newton_f
            FUNCTION DESCRIPTION: function used by the secant method 
            to find the zero and c parameter used in the sigmoidal 
            (sig) function.
            INPUT PARAMS:   1) c: parameter used to adjust the the 
            sigmoidal function so the motor unit twitch force saturates 
            at a determined motor unit discharge rate
                            2) force: The simulated motor unit force 
                            over time. 
            OUTPUT PARAMS:  1) s_max: returns the result of the function
                            used by the secant method (which should be
                            nearest zero value) """
        expfc = np.exp(-force*c)
        s_max= max((1-expfc)/(1+expfc)) - self.newton_target
        return s_max

    def get_force(self):
        self.neural_input = self.mnpool.neural_input
        self.t = self.mnpool.t
        self.LR = self.mnpool.LR
        if self.neural_input == []:
            print("Motor unit spike trains not found.")
            print(  "Please click 'Run interact' button at motor unit \
                    spike train section and reload this cell.")
        else:
            print("Processing...")
            self.gen_muscle_force(0.4)

    def view_force(self):
        self.get_force()
        clear_output()
        plt.figure(figsize = (9,4))
        plt.plot(self.t, self.muscle_force)
        plt.ylabel('Force [N]')
        plt.xlabel('Time [ms]')
            
    def gen_muscle_force(self,c_init):    
        """ FUNCTION NAME: gen_muscle_force
            FUNCTION DESCRIPTION: Generates the simulated muscle force.
            INPUT PARAMS:  1) c_init: c initial guess used in the secant
            (newton) method function """
        self.muscle_force = np.zeros(self.t.shape)
        self.mu_force     = np.zeros((len(self.neural_input),len(self.t)))
        self.c            = np.zeros(len(self.neural_input))
        for i in range(len(self.neural_input)):
            force = self.gen_mu_force(self.neural_input[i],1,self.T[i])
            self.c[i],tet_twitch = self.find_tetanic_parameters(i,c_init)
            temp                 = self.fP*self.P[i]/tet_twitch
            self.mu_force[i]     = self.sig(self.c[i],force)*temp
            self.muscle_force    += self.mu_force[i]       
        
    def analysis(self,add_rms, a_interval, add_spec, add_welch, spec_w,
                 spec_w_size, spec_ol, welch_w, welch_w_size, welch_ol,
                 add_mu_c, mu_index):
        if self.muscle_force == []:
            print(  "Muscle force not found.")
            print(  "Please click on 'Run interact' button at Force \
                    Generation section and run this cell again.")
        else:
            self.t      = self.mnpool.t
            a_init      = int(a_interval[0] * self.sampling / 1e3)
            a_end       = int(a_interval[1] * self.sampling / 1e3)
            aForce      = self.muscle_force[a_init:a_end]
            at          = self.t[a_init:a_end]
            g_count     = 0
            force_mean  = np.mean(aForce)
            if add_rms:      
                force_sd = np.std(aForce)
                plt.figure(figsize = (9, 4))
                self.plot_std(  at, aForce, force_mean, force_sd, 
                                'Muscle Force', 'Force [N]')
            if add_spec:
                if (spec_w_size <= spec_ol):
                    spec_w_size = spec_ol + 1
                spec_w_size = int(spec_w_size * self.sampling / 1e3)
                spec_window = get_window(spec_w, spec_w_size)
                f, tf, Sxx = spectrogram(
                    aForce, self.sampling, 
                    window = spec_window,
                    nperseg = spec_w_size, 
                    noverlap = int(spec_ol * self.sampling / 1e3)
                )
                plt.figure(figsize = (9,4))
                ax1=plt.subplot(111)
                self.plot_spec(tf*1e3+a_interval[0], f, Sxx, ax1, 10)
            if add_welch:
                if (welch_w_size <= welch_ol):
                    welch_w_size = welch_ol + 1
                welch_w_size     = int(welch_w_size*self.sampling/1e3)
                fwelch, PSDwelch = welch(aForce, self.sampling, 
                    window=welch_w, nperseg=welch_w_size,
                    noverlap=welch_ol*self.sampling/1e3)
                forcefm, psdfm   = self.MedFreq(fwelch, PSDwelch)
                plt.figure(figsize=(9,4))
                self.plot_welch(fwelch, PSDwelch, forcefm, psdfm, 10, 
                                "Power [N\u00b2/Hz]")
            if add_mu_c:
                plt.figure(figsize = (9,4))
                mu_c_force = self.mu_force[mu_index-1]
                a_force = mu_c_force[a_init:a_end]
                self.plot_mu_force_c(at, a_force, mu_index)
    
    def MedFreq(self,freq,psd):
        cum = cumtrapz(psd, freq, initial = 0)
        f   = np.interp(cum[-1]/2, cum, freq)
        mpf = cum[-1]/2
        return f,pfr
            
    def plot_std(self,at,aemg,mean,std,title,ylabel):
        plt.ylabel(ylabel)
        plt.xlabel('Time [ms]')
        plt.plot(at,aemg, label='Force')
        print("Mean muscle force = %.3f N" %(mean))
        print("Force standard deviation = %.5f N" %(std))
        plt.xlim(at[0],at[-1])
        
    def plot_spec(self, tf, f, Sxx, spec_axis, ylim):
        cf      = plt.contourf(tf,f,Sxx, levels=20, cmap=plt.cm.jet)
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [ms]")
        plt.ylim(0,ylim)
        ax_in   = inset_axes(spec_axis, width="5%",height="100%", loc=3, 
                           bbox_to_anchor=(1.01, 0., 1, 1), borderpad=0,
                           bbox_transform=spec_axis.transAxes)
        cbar    = plt.colorbar(cf,cax = ax_in)
        cbar.ax.set_ylabel('[$N^2$]',rotation=0, va='bottom')
        cbar.ax.yaxis.set_label_coords(0.5,1.05)
        
    def plot_welch(self,fwelch,PSDwelch,emgfm,psdfm,xlim,ylabel):
        plt.plot(fwelch,PSDwelch)
        plt.axvline(x=emgfm,ls = '--',lw=0.5, c = 'k')
        print("PSD median frequency = %.3f Hz" %emgfm)
        plt.ylabel(ylabel)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0,xlim)
        plt.ylim(bottom=0)

    def plot_mu_force_c(self,at,mu_force,mu_c_index):
        plt.plot(at,mu_force*1e3)
        plt.xlabel('Time [ms]')
        plt.ylabel('Force [mN]')
        plt.title('MU # {}'.format(mu_c_index))
            
    def save_config(self):
        try:
            self.config.update({
                    'P_1 [mN]': self.fP*1e3,
                    'R_P': self.RP,
                    'Twitch duration': self.dur_type,
                    'T_L [ms]': self.Tl,
                    'R_T':self.RT,
                    'MU_1 saturation frequency [Hz]': self.fsatf,
                    'MU_n saturation frequency [Hz]': self.lsatf,
                    'T_i gaussian mean [ms]': self.CT_mean,
                    'T_i gaussian std [ms]': self.CT_std}) 
        except:
            print(  'Could not save force parameters. Try to click on \
                    \'run interact\' button on force generation cell.')
        return self.config