import sys
import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt, welch, get_window, spectrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import clear_output

class Emg_mod(object):
    def __init__(self,mnpool):
        self.morpho     = 'Ring'               
        self.csa        = 150e-6
        self.fat        = 0.2e-3
        self.skin       = 0.1e-3
        self.theta      = 0.9
        self.prop       = 0.4
        self.first      = 21
        self.ratio      = 84
        self.t1m        = 0.7
        self.t1m_save   = 0
        self.t2m_save   = 0
        self.t1dp       = 0.5
        self.t2m        = 1
        self.t2dp       = 0.25
        self.emg        = []
        self.n          = mnpool.n
        self.t1         = mnpool.t1
        self.t2a        = mnpool.t2a
        self.t2b        = mnpool.t2b
        self.MUFN       = np.zeros(self.n)
        self.MUradius   = np.zeros(self.n)
        self.LR         = mnpool.LR
        self.t          = mnpool.t
        self.rr         = mnpool.rr
        self.v1         = 1
        self.v2         = 130
        self.d1         = 3
        self.d2         = 1
        self.expn_interpol()
        self.exp_interpol()
        self.neural_input = mnpool.neural_input
        self.sampling     = mnpool.sampling
        self.mnpool       = mnpool
        self.config       = {}
        self.mu_distance  = []
        self.add_noise    = False 
        self.noise_level  = 0
        self.add_filter   = False
        self.lowcut       = 0
        self.highcut      = 0

        
    def defineMorpho(self):
        """ FUNCTION NAME: defineMorpho
            FUNCTION DESCRIPTION: Defines the muscle morphology and electrode 
            position"""
        if self.morpho == 'Circle':
            self.circle_tissues()
        elif self.morpho == 'Ring':
            self.ring_tissues()
        elif self.morpho == 'Pizza':
            self.pizza_tissues()
        elif self.morpho == 'Ellipse':
            self.prop = 1/self.prop
            self.ellipse_tissues()

    def circle_tissues(self):
        """ FUNCTION NAME: circle_tissues
            FUNCTION DESCRIPTION: Draw Circle Tissue limits by creating arrays
            coordinates """
        self.r      = r =  np.sqrt(self.csa/np.pi)
        circle      = np.arange(0,2*np.pi,0.01) 
        self.ma     = r * np.cos(circle)
        self.mb     = r * np.sin(circle)
        self.fa     = (r+self.fat)* np.cos(circle)
        self.fb     = (r+self.fat)* np.sin(circle)
        self.sa     = (r+self.fat+self.skin)* np.cos(circle)
        self.sb     = (r+self.fat+self.skin)* np.sin(circle)
        self.elec   = r+self.fat+self.skin
        
    def ring_tissues(self):
        """ FUNCTION NAME: ring_tissues
            FUNCTION DESCRIPTION: Draw ring Tissue limits by creating arrays 
                                  coordinates of the tissue boundaries. """
        self.re     = np.sqrt((self.csa)/(self.theta*(1-self.prop**2)))
        self.ri     = self.re * self.prop
        angle       = np.arange(np.pi/2-self.theta,np.pi/2+self.theta,0.01)
        t1          = [self.ri*np.cos(np.pi/2-self.theta)]
        t2          = self.re*np.cos(angle)
        self.ma     = np.concatenate((t1,t2,np.flip(self.ri*np.cos(angle),0)))
        t1          = [self.ri*np.sin(np.pi/2-self.theta)]
        t2          = self.re*np.sin(angle)
        self.mb     = np.concatenate((t1,t2,np.flip(self.ri*np.sin(angle),0)))
        self.fa     = (self.re + self.fat) * np.cos(angle)
        self.fb     = (self.re + self.fat) * np.sin(angle)
        self.sa     = (self.re + self.fat + self.skin) * np.cos(angle)
        self.sb     = (self.re + self.fat + self.skin) * np.sin(angle)
        self.elec   = self.re + self.fat + self.skin
        
    def pizza_tissues(self):
        """ FUNCTION NAME: pizza_tissues
            FUNCTION DESCRIPTION: Draw pizza like muscle tissue limits by 
            creating arrays coordinates of the tissue boundaries """
        self.r      = np.sqrt(self.csa/self.theta)
        angle       = np.arange(np.pi/2-self.theta, np.pi/2+self.theta, 0.01)
        self.ma     = self.r * np.cos(angle)
        self.mb     = self.r * np.sin(angle)
        self.ma     = np.concatenate(([0],self.ma,[0]))
        self.mb     = np.concatenate(([0],self.mb,[0]))
        self.fa     = (self.r+self.fat)* np.cos(angle)
        self.fb     = (self.r+self.fat)* np.sin(angle)
        self.sa     = (self.r+self.fat+self.skin)* np.cos(angle)
        self.sb     = (self.r+self.fat+self.skin)* np.sin(angle)
        self.elec   = self.r+self.fat+self.skin
        
    def ellipse_tissues(self):
        """ FUNCTION NAME: ellipse_tissues
            FUNCTION DESCRIPTION: Draw ellipse like muscle tissue limits by
            creating arrays of coordinates of the tissue boundaries. """
        self.b      = np.sqrt(self.csa/(self.prop*np.pi))
        self.a      = self.prop*self.b
        circle      = np.arange(0,2*np.pi,0.01)
        self.ma     = self.a * np.cos(circle)
        self.mb     = self.b * np.sin(circle)
        self.fa     = (self.a + self.fat) * np.cos(circle)
        self.fb     = (self.b + self.fat) * np.sin(circle)
        self.sa     = (self.a + self.fat + self.skin) * np.cos(circle)
        self.sb     = (self.b + self.fat + self.skin) * np.sin(circle)
        self.elec   = self.b + self.fat + self.skin

    def view_morpho(self,CSA, prop, the, sk, fa,morpho):
        """ FUNCTION NAME: view_morpho
            FUNCTION DESCRIPTION: Plot graphic with muscle cross-sectional 
            area morphology
            INPUT:  1) CSA: muscle cross sectional area [mm²]
                    2) prop: muscle radius proportion
                    3) the: theat angle. Parameter to define muscle morphology
                    4) sk: skin tickness
                    5) fa: adipose tickness
                    6) morpho: muscle morphology """
        self.csa, self.skin, self.fat = CSA*10**-6, sk*10**-3, fa*10**-3
        self.theta, self.prop, self.morpho= the, prop , morpho
        self.defineMorpho()
        plt.figure(figsize=(5,4))
        plt.plot(self.ma*1e3, self.mb*1e3, ls='-.', label='Muscle boundaries')
        plt.plot(self.fa*1e3, self.fb*1e3, ls='--', label='Fat tissue')
        plt.plot(self.sa*1e3, self.sb*1e3, label='Skin boundaries')
        plt.plot(self.elec*1e3, marker=7, ms='15', label='Electrode')
        plt.legend(loc=10)
        plt.axis('equal')
        plt.xlabel('[mm]')
        plt.ylabel('[mm]')
        
    def innervateRatio(self):
        """ FUNCTION NAME: innervateRatio
        FUNCTION DESCRIPTION: Calc    ulates the number of innervated muscle 
        fibers for each  motorneuron in the pool and the motor unit territory
        radius. Based on the work of Enoka e Fuglevand, 2001."""
        n_fibers = 0
        for i in range(self.n):
            self.MUFN[i] = self.first*np.exp(np.log(self.ratio)*(i)/self.n)
            n_fibers     = n_fibers + self.MUFN[i]
        fiber_area    = self.csa/n_fibers
        MUarea        = self.MUFN*fiber_area
        self.MUradius = np.sqrt(MUarea/np.pi)

    # FUNCTION NAME: gen_distribution
    # FUNCTION DESCRIPTION: Defines the motor units  x and y coordinates 
    def gen_distribution(self):
        self.t1m_save = self.t1m
        self.t2m_save = self.t2m
        if self.morpho == 'Circle':
            self.t1m = self.t1m*self.r
            self.t2m = self.t2m*self.r
            self.circle_normal_distribution_otimize()
        elif self.morpho == 'Ring':
            self.t1m = self.ri + (self.re-self.ri) * self.t1m
            self.t2m = self.ri + (self.re-self.ri) * self.t2m
            flag = 1 
            while(flag == 1):
                flag = self.ring_normal_distribution_otimize()
        elif self.morpho == 'Pizza':
            self.t1m = self.t1m * self.r
            self.t2m = self.t2m * self.r
            self.pizza_normal_distribution_otimize()
        elif self.morpho == 'Ellipse':
            self.ellipse_normal_distribution_otimize()

    def circle_normal_distribution_otimize(self):
        """ FUNCTION NAME: circle_normal_distribution_otimize
            FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center 
            coordinates for circle cross sectional area (CSA) muscle 
            morpholigies. Verifies MUT placed before to otimize by reducing 
            the distribution variability across CSA. """
        self.x= np.zeros(self.n)
        self.y= np.zeros(self.n)
        i= self.n - 1
        while (i> self.t1):
            temp   = (self.r-self.MUradius[i])*self.t2dp 
            r_temp = np.random.normal(self.t2m,temp)
            t_temp = np.random.uniform(0,2*np.pi)
            x_temp = r_temp*np.cos(t_temp)
            y_temp = r_temp*np.sin(t_temp)
            if r_temp <= self.r- self.MUradius[i] and r_temp>= 0:
                if i == 0:
                    self.x[i]   = x_temp
                    self.y[i]   = y_temp
                    i           = i-1
                else:
                    ant_d = self.r
                    for j in range(i,self.n):
                        temp    = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d       = np.sqrt(temp)
                        min_d   = min(d,ant_d)
                        ant_d   = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i]   = x_temp
                        self.y[i]   = y_temp
                        i           = i - 1
        while (i>=0):
            temp   = (self.r-self.MUradius[i])*self.t1dp
            r_temp = np.random.normal(self.t1m,temp)
            t_temp = np.random.uniform(0,2*np.pi)
            x_temp = r_temp*np.cos(t_temp)
            y_temp = r_temp*np.sin(t_temp)
            if r_temp <= self.r-self.MUradius[i] and r_temp>= 0:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i         = i-1
                else:
                    ant_d = self.r
                    for j in range(i,self.n):
                        temp  = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d     = np.sqrt(temp)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i         = i - 1

    def motorUnitTerritory(self):
        """ FUNCTION NAME: motorUnitTerritory
            FUNCTION DESCRIPTION: Based on motor unit territory centers and 
            muscle radius, creates the motor unit territory boundaries."""
        theta = np.arange(0,2*np.pi,0.01)
        self.MUT = np.zeros((self.n,2,len(theta)))
        for i in range(self.n):
            self.MUT[i][0] = self.x[i] + self.MUradius[i]*np.cos(theta)
            self.MUT[i][1] = self.y[i] + self.MUradius[i]*np.sin(theta)

    def quantification_of_mu_regionalization(self):
        """ FUNCTION NAME: quantification_of_mu_regionalization
            FUNCTION DESCRIPTION: Calculates the motor unit type II territory 
            radial eccentricity. """
        n_f     = [self.n,self.n]
        n_i     = [0,self.t1]
        peso    = [0,0]
        r_cg    = [0,0]
        for j in range(2):
            for i in range (n_i[j],n_f[j]):
                peso[j] = peso[j] + self.MUFN[i]
                temp    = self.x[i]**2 + self.y[i]**2
                r_cg[j] = r_cg[j] + np.sqrt(temp)*self.MUFN[i]
            r_cg[j] = r_cg[j]/peso[j]
        self.eccentricity = r_cg[1] - r_cg[0]

    def generate_density_grid(self):
        """ FUNCTION NAME: generate_density_grid
            FUNCTION DESCRIPTION: Generates the muscle cross sectional area 
            density of motor unit territories (to use with 2d histogram) """
        self.grid_step  = 5e-5 
        self.gx         = np.zeros(1)
        self.gy         = np.zeros(1)
        for i in range(self.n):
            murad = self.MUradius[i]
            x_temp = np.arange(-murad, murad, self.grid_step)
            for j in range(len(x_temp)):
                Y       = np.sqrt(self.MUradius[i]**2-x_temp[j]**2)
                y_temp  = np.arange(-Y,Y,self.grid_step)
                x_temp2 = (x_temp[j]*np.ones(len(y_temp))+self.x[i])
                self.gx = np.append(self.gx,x_temp2)
                self.gy = np.append(self.gy,(y_temp+self.y[i]))
    
    def get_distribution(self,ratio, t1m, t1dp, t2m, t2dp):
        self.ratio, self.t1m, self.t1dp = ratio, t1m, t1dp
        self.t2m, self.t2dp = t2m, t2dp
        self.innervateRatio()
        self.gen_distribution()
        self.motorUnitTerritory()
        self.quantification_of_mu_regionalization()
        self.generate_density_grid()
        self.LR = self.mnpool.LR

    def view_distribution(self,ratio, t1m, t1dp, t2m, t2dp):
        """ FUNCTION NAME: view_distribution
            FUNCTION DESCRIPTION: Generates graphic with motor unit 
            distribution within muscle csa and 2d histogram
            INPUT:   1) ratio: ratio between first and last motor unit muscle 
            fiber quantity
                     2) t1m: Type I MU distance distribution mean (% relative 
            to the muscle radius) [float]
                     3) t1dp: Type I MU distance distribution standard 
            deviation (% relative to the muscle radius) [float]
                     4) t2m: Type II MU distance distribution mean (% relative
            to the muscle radius) [float]
                     5) t2dp: Type II MU distance distribution standard 
            deviation (% relative to the muscle radius) [float] """
        self.get_distribution(ratio, t1m, t1dp, t2m, t2dp)
        hst_step  = 0.1
        hist_bins = [np.arange(min(self.ma)*1e3, max(self.ma)*1e3, hst_step), 
                     np.arange(min(self.mb)*1e3, max(self.mb)*1e3, hst_step)]
        hist,xedges,yedges = np.histogram2d(    self.gx*1e3, self.gy*1e3, 
                                                bins=hist_bins)
        f, axes = plt.subplots(1, 2, figsize=(8, 4), sharex = 'all')
        plt.sca(axes[0])
        plt.ylabel('[mm]')
        plt.xlabel('[mm]')
        ecc2 = self.eccentricity*1e3
        print("MU type II radial eccentricity:{:.2f} [mm]".format(ecc2))
        fill_blue = mpl.patches.Patch(  label='Recruited type I MUs', 
                                        fc=(0,0,1,0.4))
        fill_red  = mpl.patches.Patch(  label='Recruited type II MUs', 
                                        fc=(1,0,0,0.4))
        blue_line = mpl.lines.Line2D(   [], [], color='b', label='Type I MU')
        red_line  = mpl.lines.Line2D(   [], [], color='r', ls='--', 
                                        label='Type II MU')
        plt.legend( handles=[fill_blue, fill_red, blue_line, red_line],
                    loc=3, fontsize=8)
        for i in range(self.t1):
            if (i <= self.LR):
                plt.fill(   self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, 
                            fc=(0,0,1,0.4), lw=0.5)
            plt.plot(self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, color='b')
        for i in range(self.t1, self.n):
            if (i <= self.LR):
                plt.fill(   self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, 
                            fc=(1,0,0, 0.4))
            plt.plot(self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, color='r', ls='--')
        plt.plot(   self.ma*1e3, self.mb*1e3, self.fa*1e3, self.fb*1e3, 
                    self.sa*1e3, self.sb*1e3)
        plt.plot(self.elec*1e3, marker=7, ms='15')
        plt.axis('equal')
        plt.sca(axes[1])
        axes1  = plt.gca()
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        with plt.style.context('ggplot'):
            im1     = plt.imshow(    hist.T, extent=extent, cmap=plt.cm.jet,
                                     interpolation='nearest', origin='lower')
            axins1  = inset_axes(axes1, width="5%", height="100%", loc=3, 
                                 bbox_to_anchor=(1.01, 0., 1, 1),
                                 bbox_transform=axes1.transAxes, borderpad=0)
            cbar    = plt.colorbar(im1, cax=axins1)
            axins1.xaxis.set_ticks_position("bottom")
            cbar.ax.set_ylabel('[a.u.]',rotation=0, va='bottom')
            cbar.ax.yaxis.set_label_coords(0.5,1.05)
            axes1.axis('equal')
            axes1.set_xlabel('[mm]')
        plt.subplots_adjust(wspace = 0.2, hspace = 0.05)
        
    def ring_normal_distribution_otimize(self):
        """ FUNCTION NAME: ring_normal_distribution_otimize
            FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center 
            coordinates for ring cross sectional area (CSA) muscle 
            morpholigies. Verifies MUT placed before to otimize by reducing 
            the distribution variability across CSA. """
        counter         = 0
        counter_limit   = 10e3
        flag            = 0
        self.x          = np.zeros(self.n)
        self.y          = np.zeros(self.n)
        i               = self.n - 1
        while (i> self.t1):
            std_temp = (self.re-self.MUradius[i])*self.t2dp
            r_temp   = np.random.normal(self.t2m,std_temp)
            t_temp   = np.random.uniform(-self.theta,self.theta)
            x_temp   = r_temp*np.sin(t_temp)
            y_temp   = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            c1 = (r_temp <= self.re-self.MUradius[i])
            c2 = (r_temp>= self.MUradius[i] + self.ri)
            c3 = (phi_c <= self.theta - theta_c)
            c4 = (phi_c >= -(self.theta - theta_c))
            if  c1 and c2 and c3 and c4:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.re
                    for j in range(i,self.n):
                        dsquare = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d       = np.sqrt(dsquare)
                        min_d   = min(d,ant_d)
                        ant_d   = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
                    else:
                        counter += 1
                        if counter > counter_limit:
                            flag = 1
                            break
        while (i>=0):
            std_temp = (self.re-self.MUradius[i])*self.t1dp
            r_temp   = np.random.normal(self.t1m,std_temp)
            t_temp   = np.random.uniform(-self.theta,self.theta)
            x_temp   = r_temp*np.sin(t_temp)
            y_temp   = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            c1 = (r_temp <= self.re-self.MUradius[i])
            c2 = (r_temp>= self.MUradius[i]+self.ri)
            c3 = (phi_c <= self.theta - theta_c)
            c4 = (phi_c >= -(self.theta - theta_c))
            if c1 and c2 and c3 and c4:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.re
                    for j in range(i,self.n):
                        dsquare = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d       = np.sqrt(dsquare)
                        min_d   = min(d,ant_d)
                        ant_d   = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
                    else:
                        counter += 1
                        if counter > counter_limit:
                            flag = 1
                            break
        return flag
                        
    # FUNCTION NAME: pizza_normal_distribution_otimize
    # FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center coordinates for
    #                       pizza like cross sectional area (CSA) muscle morpholigies. 
    #                       Verifies MUT placed before to otimize by reducing the 
    #                       distribution variability across CSA.
    def pizza_normal_distribution_otimize(self):
        self.x= np.zeros(self.n)
        self.y= np.zeros(self.n)
        i= self.n - 1
        while (i> self.t1):
            r_temp = np.random.normal(self.t2m,(self.r-self.MUradius[i])*self.t2dp)
            t_temp = np.random.uniform(-self.theta,self.theta)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            if (r_temp <= self.r-self.MUradius[i]) and (r_temp >= self.MUradius[i]) \
                and (phi_c <= self.theta - theta_c) and (phi_c >= -(self.theta - theta_c)):
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.r
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        while (i>=0):
            #r_temp = r*np.random.uniform()
            r_temp = np.random.normal(self.t1m,(self.r-self.MUradius[i])*self.t1dp)
            t_temp = np.random.uniform(-self.theta,self.theta)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            if (r_temp <= self.r-self.MUradius[i]) and (r_temp>= self.MUradius[i]) \
                and (phi_c <= self.theta - theta_c) and (phi_c >= -(self.theta - theta_c)):
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.r
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
                        
    # FUNCTION NAME: ellipse_normal_distribution_otimize
    # FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center coordinates for
    #                       pizza like cross sectional area (CSA) muscle morpholigies. 
    #                       Verifies MUT placed before to otimize by reducing the 
    #                       distribution variability across CSA.
    def ellipse_normal_distribution_otimize(self):
        self.x= np.zeros(self.n)
        self.y= np.zeros(self.n)
        i= self.n - 1
        while (i> self.t1):
            t_temp = np.random.uniform(0,2*np.pi)
            raio = self.a * self.b/np.sqrt((self.b*np.cos(t_temp))**2+(self.a*np.sin(t_temp))**2)
            r_temp = np.random.normal(raio*self.t2m,(raio-self.MUradius[i])*self.t2dp)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if r_temp <= raio-self.MUradius[i] and r_temp >= 0:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = raio
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        while (i>=0):
            t_temp = np.random.uniform(0,2*np.pi)
            raio = self.a*self.b/np.sqrt((self.b*np.cos(t_temp))**2+(self.a*np.sin(t_temp))**2)
            r_temp = np.random.normal(raio*self.t1m,(raio-self.MUradius[i])*self.t1dp)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if r_temp <= raio-self.MUradius[i] and r_temp >= 0:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = raio
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min =self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        temp = self.x
        self.x = self.y
        self.y = temp

    def get_MUAPs(self):
        comp = max(self.lmvar) / (1.8)
        x = np.arange(0, 12 * comp, 0.1)
        z = np.zeros((self.LR,len(x)))
        for i in range(self.LR):
            if (self.hr_order[i] == 1):
                z[i] = self.hr1_f(x,self.ampvar[i],self.lmvar[i],[max(self.lmvar) * 3])
            else:
                z[i] = self.hr2_f(x,self.ampvar[i],self.lmvar[i],[max(self.lmvar) * 3])
        return x,z

    def get_ptp(self):
        x,muaps = self.get_MUAPs()
        ptp_durs = np.zeros(self.LR)
        ptp_amps = np.zeros(self.LR)
        for i in range(self.LR):
            min_val = min(muaps[i])
            index_min = np.where(muaps[i] == min_val)[0][0]
            max_val = max(muaps[i])
            index_max = np.where(muaps[i] == max_val)[0][0]
            ptp_durs[i] = abs(x[index_min]-x[index_max])
            ptp_amps[i] =  max_val - min_val
        return ptp_amps,ptp_durs

    def view_muap(self,v1, v2, d1, d2, add_hr):
        """    # FUNCTION NAME: view_muap
        # FUNCTION DESCRIPTION: plot graphic with motor unit action potential
        #                INPUT: 1) v1: first recruited muap amplitude factor
        #                       2) v2: last recruited muap amplitude factor
        #                       3) d1: first recruited muap duration factor
        #                       4) d2: last recruited muap duration factor
        #                       5) add_hr: Visualize 2nd order hermite rodrigues(MUAP) function"""
        self.v1, self.v2, self.d1, self.d2 = v1, v2, d1, d2
        self.v2 = v2
        self.d1 = d1
        self.expn_interpol()
        self.exp_interpol()
        
        comp = max(self.lm_array) / (1.8)
        x = np.arange(0, 12 * comp, 0.1)
        y = np.arange(self.n)
        z = np.zeros((y.shape[0],x.shape[0]))
        
        fig,ax = plt.subplots(3,1,figsize=(5,9))
        plt.sca(ax[0])
        for i in range(self.n):
            if (add_hr == '1st order'):
                z[i] = self.hr1_f(x,self.amp_array[i],self.lm_array[i],[max(self.lm_array) * 3])
            else:
                z[i] = self.hr2_f(x,self.amp_array[i],self.lm_array[i],[max(self.lm_array) * 3])
            if i % int(self.n/5) == 0:
                plt.plot(x,z[i], label = 'MU${}_{%d}$'%(i+1)) 
        plt.plot(x,z[-1], label = 'MU${}_{%d}$'%(i+1)) 
                
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.legend(loc=1)
        plt.xlim(left=0)

        plt.sca(ax[1])
        plt.plot(np.linspace(1,self.n,self.n), self.amp_array)
        plt.xlabel('MU $i$')
        plt.ylabel('$A_M$ [ms]')
        plt.xlim(1,self.n)
        plt.ylim(bottom =0)

        plt.sca(ax[2])
        plt.plot(np.linspace(1,self.n,self.n), self.lm_array)
        plt.xlabel('MU $i$')
        plt.ylabel('$\\lambda_M$ [mV]')
        plt.xlim(1,self.n)

        plt.tight_layout()


    # FUNCTION NAME: exp_interpol
    # FUNCTION DESCRIPTION: Creates an  growing exponential interpolation
    #                        with n points  between  two values (v1 and v2)
    #                        adjusted by the rr factor.
    def exp_interpol(self):
        a=np.log(self.rr)/self.n
        rte = np.zeros(self.n)
        self.amp_array = np.zeros(self.n)
        for i in range(self.n):
            rte[i]=np.exp(a*(i+1))
            self.amp_array[i] = rte[i]*(self.v2-self.v1)/self.rr + self.v1

    # FUNCTION NAME: expn_interpol
    # FUNCTION DESCRIPTION: Creates an descending exponential interpolation
    #                        with n points  between  two values (d1 and d2)
    #                        adjusted by the rr factor.
    def expn_interpol(self):      
        a = np.log(self.rr)/self.n
        rte = np.zeros(self.n)
        self.lm_array = np.zeros(self.n)
        for i in range(self.n):
            rte[i]= np.exp(-a*(i+1))
            self.lm_array[i] = rte[i]*(self.d1-self.d2)+self.d2

    # FUNCTION NAME: hr1_f
    # FUNCTION DESCRIPTION: hermite-rodriguez 1nd order function (Cisi e Kohn, 2008)
    # INPUT PARAMS:  1) t: time simulation array (ms) [numpy array]
    #                2) i: motor unit index
    #                3) tspk: Motorneuron discharge times (ms) [List of floats]
    # OUTPUT PARAMS: 1) hr1: biphasic motor unit action potential train over time 
    #                   (mV) [numpy array]
    def hr1_f(self,t,amp,lm,tspk):
        n = len(t)
        hr1= np.zeros(n)
        j=0
        sbase = lm*3
        for w in range(n):
            if (t[w] > tspk[j]+ sbase) and (j < len(tspk)-1) and j < len(tspk):
                j = j+1
            #hr1(i) =   first order Hermite-Rodriguez in instant 't'
            hr1[w] = amp*((t[w]-tspk[j])/lm)*np.exp(-1*(((t[w]-tspk[j])/lm)**2))
        return hr1

    # FUNCTION NAME: hr2_f
    # FUNCTION DESCRIPTION: hermite-rodriguez 2nd order function (Cisi e Kohn, 2008)
    # INPUT PARAMS:  1) t: time simulation array (ms) [numpy array]
    #                2) i: motor unit index
    #                3) tspk: Motorneuron discharge times (ms) [List of floats]
    # OUTPUT PARAMS: 1) hr2: triphasic motor unit action potential train 
    #                   over time (mV) [numpy array]
    def hr2_f(self,t,amp,lm,tspk):
        n = len(t)
        hr2= np.zeros(n)
        j=0
        sbase = lm*3
        for w in range(n):
            if t[w] > tspk[j]+ sbase and j<len(tspk)-1 and j < len(tspk):
                j = j+1
            #hr2(i) =  Second order Hermite-Rodriguez in instant 't'
            hr2[w] = amp*(1-2*((t[w]-tspk[j])/lm)**2) \
                    *np.exp(-1*(((t[w]-tspk[j])/lm)**2))
        return hr2
    

    # FUNCTION NAME: vc_filter
    # FUNCTION DESCRIPTION: Apply the filtering effect of the muscle tissue 
    #                       (volume conductor) on the duration and amplitude
    #                       factors of the hermite rodriguez functions
    def vc_filter(self):
        self.mu_distance = np.sqrt((self.elec-self.y)**2+self.x**2)
        self.ampvar = self.amp_array*np.exp(-self.mu_distance/self.ampk)
        self.lmvar = self.lm_array*(1+self.durak*self.mu_distance)
    
    # FUNCTION NAME: view_attenuation
    # FUNCTION DESCRIPTION: generates graphic plot of the volume conduction attenuation
    # INPUT PARAMS:  1) ampk: Volume conductor amplitude attenuation constant 
    #                2) durak: Volume conductor duration widening constant 
    def view_attenuation(self,ampk,durak):
        self.ampk = ampk*1e-3
        self.durak = durak*1e3
        step = 1e-4
        ga = np.arange(min(self.ma), max(self.ma), step)
        gb = np.arange(min(self.mb), max(self.mb), step)
        Ga, Gb = np.meshgrid(ga, gb)
        mu_distance2d = np.sqrt((self.elec-Gb) ** 2 + Ga ** 2)
        apvar2d = np.exp(-mu_distance2d / self.ampk)
        lmvar2d = 1 + self.durak * mu_distance2d
        f = plt.figure(figsize = (9,4))
        axes1 = plt.subplot(121)
        axes1.axis('equal')
        plt.xlabel('[mm]')
        plt.ylabel('[mm]')
        
        plt.plot(self.ma * 1e3, self.mb * 1e3, self.fa * 1e3, self.fb * 1e3, self.sa * 1e3, self.sb *1e3)
        plt.plot(self.elec * 1e3, marker = 7, ms = '15')
        CS = plt.contour(Ga * 1e3, Gb * 1e3, apvar2d, 10, cmap=plt.cm.jet_r)
        ax1= plt.gca()
        cbar1 = plt.colorbar(CS, ax = ax1)
        cbar1.ax.set_ylabel('Attenuation',rotation=0,va='bottom')
        cbar1.ax.yaxis.set_label_coords(0.5,1.05)
        axes1.clabel(CS, inline = 1, fontsize = 12)
        ax2 = plt.subplot(122)
        plt.plot(self.ma * 1e3, self.mb * 1e3, self.fa * 1e3, self.fb * 1e3, self.sa * 1e3, self.sb * 1e3)
        plt.plot(self.elec * 1e3, marker = 7, ms = '15')
        CS2 = plt.contour(Ga * 1e3, Gb * 1e3, lmvar2d, 8, cmap=plt.cm.jet)
        ax2 = plt.gca()
        cbar2 = plt.colorbar(CS2, ax = ax2)
        cbar2.ax.set_ylabel('Widening',rotation=0,va='bottom')
        cbar2.ax.yaxis.set_label_coords(0.5,1.05)
        ax2.clabel(CS2, inline = 1, fontsize = 12)
        ax2.axis('equal')
        plt.xlabel('[mm]')
        f.subplots_adjust(wspace=0.2)
        
    def get_semg(self,add_noise, noise_level, add_filter, bplc, bphc):
        self.add_noise, self.noise_level, self.add_filter = add_noise, noise_level, add_filter
        self.lowcut, self.highcut   = bplc, bphc
        self.neural_input           = self.mnpool.neural_input
        self.LR                     = self.mnpool.LR
        self.t                      = self.mnpool.t
        if self.neural_input == []:
            print("Neuronal drive to the muscle not found.")
            print("Please click on the button 'run interact' in the motor unit spike trains section.")
        else:
            print("Processing...")
            self.vc_filter()
            self.semg()
            if add_noise:
                self.emg = self.raw_emg  + np.random.normal(0, noise_level, len(self.raw_emg))
            else:
                self.emg = self.raw_emg
            if add_filter:
                self.butter_bandpass_filter()

    # FUNCTION NAME: view_semg
    # FUNCTION DESCRIPTION: generates and plot surface emg
    # INPUT PARAMS:  1) add_noise: add noise to the surface emg [boolean]
    #                2) noise_level: standard deviation of the noise level
    #                3) add_filter: If true, filters the emg with a butterworth 4order filter
    #                4) bplc: Butterworth bandpass low cut frequency (Hz) [float]
    #                5) bphc: Butterworth bandpass high cut frequency (Hz) [float]
    def view_semg(self,add_noise, noise_level, add_filter, bplc, bphc):
        bins = 50
        self.get_semg(add_noise, noise_level, add_filter, bplc, bphc)
        amps,durs = self.get_ptp()
        clear_output()
        fig,ax = plt.subplots(3,1,figsize = (9, 9))
        ax[0].set_ylabel("Amplitude [mV]")
        ax[0].set_xlabel('Time [ms]')
        ax[0].plot(self.t, self.emg, lw = 0.5)
        ax[0].set_xlim(0, self.t[-1])
        
        ax[1].hist(amps,bins=bins)
        ax[1].set_xlabel('MUAP amplitude [mV]')
        ax[1].set_ylabel('Count')
        
        ax[2].hist(durs,bins=bins)
        ax[2].set_xlabel('MUAP duration [ms]')
        ax[2].set_ylabel('Count')
        plt.tight_layout()
        fig.align_ylabels()
        
       


    # FUNCTION NAME: butter_bandpass_filter
    # FUNCTION DESCRIPTION: Apply Butterworth bandpass filter on data 
    def butter_bandpass_filter(self,order=4):
        nyq = 0.5 * self.sampling
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.emg = filtfilt(b, a, self.emg)
    
    # FUNCTION NAME: semg
    # FUNCTION DESCRIPTION: Generates the surface EMG signal
    def semg(self): 
        self.raw_emg    = np.zeros(len(self.t))
        self.mu_emg     = np.zeros((self.LR,len(self.t)))
        self.hr_order   = np.zeros(self.LR) 
        for i in range (self.LR):
            if np.random.randint(2) == 1: #
                temp = self.hr2_f(self.t,self.ampvar[i],self.lmvar[i],self.neural_input[i])
                self.hr_order[i] = 2
            else:
                temp = self.hr1_f(self.t,self.ampvar[i],self.lmvar[i],self.neural_input[i])
                self.hr_order[i] = 1
            self.mu_emg[i] = temp
            self.raw_emg = self.raw_emg + temp

    # FUNCTION NAME: MedFreq
    # FUNCTION DESCRIPTION: Calculates the median frequency and the power of the 
    #                       median frequency of a Power spectral density data
    # INPUT PARAMS:  1) freq: frequency vector generated with the PSD
    #                2) psd: Power spectrum density
    def MedFreq(self,freq,psd):
        cum = cumtrapz(psd,freq,initial = 0)
        f = np.interp(cum[-1]/2,cum,freq)
        mfpsdvalue = cum[-1]/2
        return f,mfpsdvalue
        
    def analysis(self,a_interval, add_rms, add_spec, add_welch, rms_length, spec_w, spec_w_size, 
                 spec_ol, welch_w, welch_w_size, welch_ol, add_mu_cont, mu_c_index):
        self.t = self.mnpool.t
        self.sampling = self.mnpool.sampling
        self.dt = self.mnpool.sampling
        if self.emg == []:
            print("Surface EMG not found.")
            print("Please click the 'Run interact' button in Surface EMG generation section and run this cell again.")
        else:
            a_init = int(a_interval[0] * self.sampling / 1e3)
            a_end =int(a_interval[1] * self.sampling / 1e3)
            aemg = self.emg[a_init:a_end]
            at = self.t[a_init:a_end]
            g_count = 0
            if add_rms:
                rms_length = int(rms_length * self.sampling/1e3)
                moving_average = np.sqrt(np.convolve(np.power(aemg,2), np.ones((rms_length,)) / rms_length, mode = 'same'))
                plt.figure(figsize = (9, 4))
                self.plot_rms(at, aemg, moving_average, 'Surface EMG Moving Average',
                              "Amplitude [mV]")
            if add_spec:
                if (spec_w_size <= spec_ol):
                    spec_w_size = spec_ol + 1
                spec_w_size = int(spec_w_size*self.sampling / 1e3)
                spec_ol = int(spec_ol*self.sampling / 1e3)
                spec_window = get_window(spec_w, spec_w_size)
                f,tf,Sxx = spectrogram(aemg, self.sampling, window = spec_window, 
                                       nperseg = spec_w_size, noverlap = spec_ol)
                plt.figure(figsize = (9, 4))
                ax1 = plt.subplot(111)
                self.plot_spec(tf * 1e3 + a_interval[0], f, Sxx, ax1, 300)
            if add_welch:
                if (welch_w_size <= welch_ol):
                    welch_w_size = welch_ol + 1
                welch_w_size = int(welch_w_size * self.sampling / 1e3)
                welch_ol = int(welch_ol * self.sampling / 1e3)
                fwelch, PSDwelch = welch(aemg * 1e-3, self.sampling, window = welch_w, 
                                         nperseg = welch_w_size, noverlap = welch_ol)
                emgfm, psdfm = self.MedFreq(fwelch, PSDwelch)
                plt.figure(figsize = (9, 4))
                self.plot_welch(fwelch, PSDwelch, emgfm, psdfm, 500, "Power [mV\u00b2/Hz]")
            if add_mu_cont:
                plt.figure(figsize = (9, 4))
                self.plot_mu_cont(at, self.mu_emg[mu_c_index - 1][a_init:a_end], mu_c_index)
        
    def plot_rms(self,at,aemg,moving_average,title,ylabel):
        plt.ylabel(ylabel)
        plt.xlabel('Time [ms]')
        #plt.title(title)
        plt.plot(at,aemg,lw=0.5,label='Raw sEMG')
        plt.plot(at,moving_average,label='Moving RMS',lw=2,color='red')
        #plt.annotate("EMG RMS = %.3f mV" %(np.sqrt(np.mean(np.square(aemg)))), xy=(0.1,0.90), xycoords = ("axes fraction"))
        print("sEMG RMS = %.3f mV" %(np.sqrt(np.mean(np.square(aemg)))))
        plt.legend(loc=1)
        plt.xlim(at[0],at[-1])
        
    def plot_spec(self,tf,f,Sxx,spec_axis,ylim):
        cf=plt.contourf(tf,f,Sxx, levels = 20, cmap=plt.cm.jet)
        #plt.title('Spectrogram')
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [ms]")
        plt.ylim(0,ylim)
        ax_in = inset_axes(spec_axis, width="5%",height="100%", loc=3, bbox_to_anchor=(1.01, 0., 1, 1),
                            bbox_transform=spec_axis.transAxes, borderpad=0)
        cbar = plt.colorbar(cf,cax = ax_in)
        ax_in.xaxis.set_ticks_position("bottom")
        cbar.ax.set_ylabel('[$mV^2$]',rotation=0, va='bottom')
        cbar.ax.yaxis.set_label_coords(0.5,1.05)
        
    def plot_welch(self,fwelch,PSDwelch,emgfm,psdfm,xlim,ylabel):
        #plt.title("sEMG Power Spectrum Density")
        plt.plot(fwelch,PSDwelch*10**6)
        plt.axvline(x=emgfm,ls = '--',lw=0.5, c = 'k')
        #plt.annotate("Median Freq. = %.2f Hz" %emgfm, xy=(0.5,0.9), xycoords = ("axes fraction"))
        print("PSD median Frequency = %.2f Hz" %emgfm)
        plt.ylabel(ylabel)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0,xlim)
        plt.ylim(bottom=0)

    def plot_mu_cont(self,at,mu_emg,mu_c_index):
        plt.plot(at,mu_emg)
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.title('MU # {}'.format(mu_c_index))
        
    def save_config(self):
        try:
            self.config.update({'CSA Morphology': self.morpho, 
                                'CSA[m^2]':  self.csa,
                                'Skin Layer [mm]': round( self.skin*1e3,6),
                                'Fat Layer [mm]':  self.fat*1e3,
                                'Proportion':  self.prop,
                                'Theta [rad]':  self.theta,
                                'Type I MU mu':  self.t1m_save,
                                'Type I MU sigma': self.t1dp,
                                'Type II MU mu':  self.t2m_save,
                                'Type II MU sigma':  self.t2dp,
                                'R_in':  self.ratio,
                                'MU_1 A_M [mV]': self.v1,
                                'MU_n A_M [mV]':  self.v2,
                                'MU_1 lambda_M [ms]': self.d1,
                                'MU_n lambda_M [ms]': self.d2,
                                'tau_at': self.ampk*1e-3,
                                'C': self.durak*1e3,
                                'Add noise':self.add_noise, 
                                'Noise std [mV]':self.noise_level,
                                'Add filter':self.add_filter,
                                'High-pass cutoff frequency [Hz]':self.lowcut,
                                'Low-pass cutoff frequency [Hz]':self.highcut})
        except AttributeError as error:
            print("Oops!", sys.exc_info()[0], "occurred: ", error)
            print(  'Could not save EMG parameters in configuration file. \
                    Try to click on \'run interact\' button on EMG \
                    generation cell.')
        return self.config