import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

class Part3():
	def __init__(self, wav0):
		self.sigmaT = 6.648e-25
		self.kerg = 1.38e-16 # Boltzmann constant erg K
		self.kev =8.61734e-5 # ev/deg
		self.hp = 6.62607e-27 # Plack constant
		self.m_e = 9.109390e-28 # electron mass
		self.c = 2.99792e10 # cm*s^-1 speed of light
		self.wav0 = wav0

	def plot_quantities(self, x, y, style='',plot_label='_nolabel',x_label='',y_label='',x_lim=None,y_lim=None, logAxes=False,logX=False,logY=False,hold=False):
			plt.rc('text',usetex=True)
			plt.rc('font',**{'family':'serif','size':14})
			if y.ndim> 1: # if x is a array of arrays
				if plot_label=='_nolabel': [plt.plot(x, y[i],style) for i in range(len(y[:,0]))]
				if isinstance(style,list)==True: [plt.plot(x, y[i], style[i],label=plot_label[i]) for i in range(len(y[:,0]))]
				else:   [plt.plot(x, y[i], style,label=plot_label[i]) for i in range(len(y[:,0]))]
				#if planet_label !='_nolegend_': plt.legend()
			else:   plt.plot(x,y,style,label=plot_label)
			plt.ylabel(y_label)
			plt.xlabel(x_label) 
			if logAxes ==True:
				plt.yscale('log')
				plt.xscale('log')
			if logY ==True: plt.yscale('log')
			if logX ==True: plt.xscale('log')
			plt.tight_layout()
			if hold == True: return
			plt.legend()
			if x_lim !=None: plt.xlim(x_lim)
			if y_lim !=None: plt.ylim(y_lim)
			plt.show()

	def planck(self,T,wav):
		h,c,kerg=self.hp,self.c,self.kerg
		#return 2*h*wav**3/c**2/(np.exp(h*wav/(kerg*T))-1.)
		return 2*h*c**2/wav**5/(np.exp(h*c/(wav*kerg*T))-1) 

	def exthmin(self,wav,temp,eldens):
		# H-minus extinction, from Gray 1992
		# input:
		# wav = wavelength [Angstrom] (float or float array)
		# temp = temperature [K]
		# eldens = electron density [electrons cm-3]    
		# output:
		# H-minus bf+ff extinction [cm^2 per neutral hydrogen atom]
		# assuming LTE ionization H/H-min
		# physics constants in cgs (all cm)
		k,h,c=self.kerg,self.hp,self.c

		# other parameters
		theta=5040./temp
		elpress=eldens*k*temp
		# evaluate H-min bound-free per H-min ion = Gray (8.11)
		# his alpha = my sigma in NGSB/AFYC (per particle without stimulated)
		sigmabf = (1.99654 -1.18267E-5*wav +2.64243E-6*wav**2 -4.40524E-10*wav**3 
			+3.23992E-14*wav**4 -1.39568E-18*wav**5 +2.78701E-23*wav**6)
		sigmabf *= 1E-18 # cm^2 per H-min ion
		if np.size(wav) > 1:
			sigmabf[np.where(wav > 16444)] = 0 # H-min ionization limit at lambda=1.6444 micron
		elif (np.size(wav) == 1):
			if wav > 16444:
				sigmabf=0
		# convert into bound-free per neutral H atom assuming Saha = Gray p135
		# units: cm2 per neutral H atom in whatever level (whole stage)
		graysaha=4.158E-10*elpress*theta**2.5*10.**(0.754*theta) # Gray (8.12)
		kappabf=sigmabf*graysaha # per neutral H atom
		kappabf=kappabf*(1.-np.exp(-h*c/(wav*1E-8*k*temp))) # correct stimulated
		# check Gray's Saha-Boltzmann with AFYC (edition 1999) p168
		# logratio=-0.1761-np.log10(elpress)+np.log10(2.)+2.5*np.log10(temp)-theta*0.754
		# print 'Hmin/H ratio=',1/(10.**logratio) # OK, same as Gray factor SB2
		# evaluate H-min free-free including stimulated emission = Gray p136
		lwav=np.log10(wav)
		f0 = -2.2763 -1.6850*lwav +0.76661*lwav**2 -0.0533464*lwav**3
		f1 = 15.2827 -9.2846*lwav +1.99381*lwav**2 -0.142631*lwav**3
		f2 = (-197.789 +190.266*lwav -67.9775*lwav**2 +10.6913*lwav**3 -0.625151*lwav**4)
		ltheta=np.log10(theta)
		kappaff = 1E-26*elpress*10**(f0+f1*ltheta+f2*ltheta**2) # Gray (8.13)
		return kappabf+kappaff

	def partfunc_Na(self,temp):
		# partition functions Na
		# input: temp (K)
		# output: float array(3) = partition functions U1,U2,U3
		if isinstance(temp,(int,float)) != True:
			u=np.zeros((3,len(temp)))
		else: u=np.zeros(3)
		# partition function Na I: follow Appendix D of Gray 1992
		# log(U1(T)) = c0 + c1 * log(theta) + c2 * log(theta)^2 +
		# c3 *log(theta)^3 + c4 log(theta)^4
		# with theta=5040./T
		theta=5040./temp
		# partition function Na I : Appendix D of Gray (1992)   
		c0=0.30955
		c1=-0.17778 
		c2=1.10594
		c3=-2.42847
		c4=1.70721
		logU1 = (c0 + c1 * np.log10(theta) + c2 * np.log10(theta)**2 +
		c3 * np.log10(theta)**3 + c4 * np.log10(theta)**4)
		u[0]=10**logU1
		# partition function Na II and Na III: approximate by the
		# statistical weights of the ion ground states
		u[1]=1 # from Allen 1976
		u[2]=6 # from Allen 1976
		return u

	def saha_Na(self,T,eldens,r):
		chiion = np.array([5.139,47.29,71.64]) #eV
		kerg,kev,h,m_e = self.kerg,self.kev,self.hp,self.m_e
		kevT =kev*T
		kergT = kerg*T
		#el_density = el_pressure/kergT
		U = self.partfunc_Na(T)
		U = np.append(U,2) # add estimated fifth value to get N_4
		sahaconst = 2*(2.*np.pi*m_e*kergT/h**2)**(3./2)/eldens
		Nstage = np.zeros(4)
		Nstage[0]=1.

		for i in range(len(chiion)):
			Nstage[i+1]=Nstage[i]*sahaconst*U[i+1]/U[i]*np.exp(-chiion[i]/kevT)

		N_total = np.sum(Nstage)
		Nr_div_Ntot = Nstage/float(N_total)
		if isinstance(r,(int,float)) ==True: # if only given a single ionization stage
			r=int(r)
			return Nr_div_Ntot[r-1] # ion stage begins at 1, not 0
		else:   # if we want all the ionization stages
			return Nr_div_Ntot


	def boltz_Na(self,T,r,s):
		c,hp,kev =self.c,self.hp,self.kev # Boltzmann constant in eV/deg
		erg2eV=1/1.60219e-12 # erg to eV conversion
		chilevel = np.array([0,hp*c/5895.94e-8*erg2eV,hp*c/5889.97e-8*erg2eV])
		g = np.array([2.,2.,4]) # 2J+1
		U = self.partfunc_Na(T)
		n_div_Nr = g[s-1]/U[r-1]*np.exp(-chilevel[s-1]/(kev*T))
		return n_div_Nr
		
	def sahabolt_Na(self,T,eldens,r,s):
		""" Saha-Boltzmann distribution for Natrium"""
		return self.saha_Na(T,eldens,r)*self.boltz_Na(T,r,s)
	
	def dopplerwidth(self,wav,T,vt):
		m = 22.99*1.6606e-24 #g
		kerg,c=self.kerg,self.c
		return wav/self.c*np.sqrt(2*kerg*T/m + vt**2)

	def voigt(self,gamma,x):
		z = (x+1j*gamma)
		V = special.wofz(z).real
		return V

	def gammavdw(self,temp,pgas,s):
		# Van der Waals broadening for Na D1 and Na D2
		# s=2 : Na D1
		# s=3 : Na D2
		# using classical recipe by Unsold
		# following recipe in SSB
		rsq_u = self.rsq_NaD(s)
		rsq_l = self.rsq_NaD(1) # lower level D1 and D2 lines is ground state s=1
		loggvdw=6.33 + 0.4*np.log10(rsq_u - rsq_l) + np.log10(pgas) - 0.7 * np.log10(temp)
		return 10**loggvdw

	def rsq_NaD(self,s):
		# compute mean square radius of level s of Na D1 and Na D2 transitions
		# -> needed for van der Waals broadening in SSB
		# s=1 : ground state, angular momentum l=0
		# s=2 : Na D1 upper level l=1
		# s=3 : Na D2 upper level l=1
		h=6.62607e-27 # Planck constant (erg s)
		c=2.99792e10 # light speed [cm/s]
		erg2eV=1/1.60219e-12 # erg to eV conversion
		E_ionization = 5.139 # [eV] ionization energy
		E_n=np.zeros(3) # energy level: E_n[0]=0 : ground state
		E_n[1]=h*c/5895.94e-8 * erg2eV # Na D1: 2.10285 eV
		E_n[2]=h*c/5889.97e-8 * erg2eV # Na D2: 2.10498 eV
		Z=1. # ionization stage, neutral Na: Na I
		Rydberg=13.6 # [eV] Rydberg constant
		l=[0.,1.,1.] # angular quantum number
		nstar_sq = Rydberg * Z**2 / (E_ionization - E_n[s-1])
		rsq=nstar_sq / 2. / Z**2 * (5*nstar_sq + 1 - 3*l[s-1]*(l[s-1] + 1))
		return rsq
		# Plot the Boltzmann and Saha distributions for checking that
		# you are at the right track


	def NaD1_ext(self,wav,T,ne,nH,vt,pgas,fudge=0.):
		wav0, h, me, c, kerg = self.wav0, self.hp, self.m_e, self.c, self.kerg
		s = 2
		r = 1
		flu = 0.318 #Oscillator strength for NaI D 1 (0.631 for NaI D 2)
		#flu = 0.631 #NaI D1
		bl = bu = 1.
		e = 4.803204e-10 #statcoulomb
		AN = 1.8e-6 #Na abundance
		nNa = AN*nH
		nlte = self.sahabolt_Na(T,ne,r,1) #Ground state
		#wav0 = 5895.94e-8    #lambda0 [cm] is center wavelength of Na D lines
		#wav0 = 5895.92e-8
		lambdaD = self.dopplerwidth(wav,T,vt)
		gamma = self.gammavdw(T,pgas,s)
		if fudge == 1.:
			gamma = 2*self.gammavdw(T,pgas,s)
		elif fudge ==2.: 
			gamma = 10**(-7.526)*nH

		v_voigt = (wav-wav0)/lambdaD
		a_voigt = wav**2/(4*np.pi*c)*gamma/lambdaD
		voigt_NaD = self.voigt(a_voigt, v_voigt) / lambdaD

		const=np.sqrt(np.pi)*e**2/(me*c)*bl/c*nNa*flu
		NaD1extinction = const*nlte*wav**2*voigt_NaD*(1-np.exp(-h*c/(wav*kerg*T)))
		#NaD1extinction = np.sqrt(np.pi)*e**2/(me*c)*wav**2/c*nlte*nNa*flu*voigt_NaD*(1-np.exp(-h*c/(wav*kerg*T)))

		#print "gamma",gamma
		#print "SahaBoltz:", nlte
		#print "wav:", wav
		#print "voigt_NaD", voigt_NaD

		#voigt_NaD REALLY big!
		return NaD1extinction

def calspectra(P,fudge,phspect,wavair, wav0, h, T, vt, ne, nH, nprot, pgas):
	sigmaT = 6.648e-25 # cm^2 Thomson scattering
	nHneutral=nH-nprot

	#----ADOPTED WAVELENGTH SPECTER----------
	#nw=1000
	#offs=np.linspace(-2e-8,2e-8,num=nw)  # brute force with very dense wavelength grid
	# bit more intelligent: dense sampling in core, sparser in wings and even sparser far wings
	l0=np.linspace(.01,.25,num=25)
	l1=np.linspace(l0[-1]+.01,1,num=10)
	l2=np.linspace(l1[-1]+.1,2,num=10)
	# make one array:
	offs=np.concatenate((np.flipud(-l2),np.flipud(-l1),np.flipud(-l0),[0],l0,l1,l2))*1e-8
    
	wav=offs+wav0
	nw=len(offs)
	int_calc = np.zeros(nw)
	nh=len(h)
	if isinstance(fudge, (int,float)): fudge=np.array([fudge])
	spectra = np.zeros((len(fudge),nw))
	for k in range(len(fudge)):
		for w in range(nw):
			wl=wav[w]
			ext=np.zeros(nh)
			tau=np.zeros(nh)
			integrand=np.zeros(nh)
			intt=0.
			for i in range(nh):
				cext = P.exthmin(wl*1e8, T[i], ne[i])*nHneutral[i] + sigmaT*ne[i]
				lext = P.NaD1_ext(wl, T[i], ne[i], nH[i], vt[i]*1e5, pgas[i],fudge[k])
				ext[i]  = cext + lext
				tau[i] = tau[i-1] + 0.5 * (ext[i] + ext[i-1]) * (h[i-1]-h[i])*1E5
				integrand[i] = P.planck(T[i],wl)*np.exp(-tau[i])
				intt += 0.5*(integrand[i]+integrand[i-1])*(tau[i]-tau[i-1])
			int_calc[w]=intt
			spectra[k,w]=intt
		spectra[k]/=np.max(spectra[k])
	
	# Return the computed spectra
	return spectra, wav

def main():
	#------------------- unpack some values
	h, T, vt, nH, nprot, ne, Ptot, PgasPtot = np.loadtxt("falc.dat",usecols=(0,3,4,5,6,7,8,9), unpack=True)
	wavnr, tellspect, phspect, phspectCor = np.loadtxt("int_nad.dat",usecols=(0,1,2,3), unpack=True)

	#------- define some constants
	sigmaT = 6.648e-25
	nHneutral=nH-nprot	# neutral hydrogen density
	pgas = Ptot*PgasPtot
	wavvac = 1./wavnr*1e8 # vacuum wavelengths in Angstrom
	wavair = 0.99972683*wavvac + 0.0107 - 196.25/wavvac	# air wavelengths

	# Find minima/ central line core wavelength of NaI D lines
	minidx1 = np.argmin(phspect)
	minidx2 = np.argmin(phspect[:minidx1-10])

	print 'vac wavelength N1 minimum =', wavvac[minidx1],'AA, ', wavvac[minidx2], 'AA'
	#---------The air wavelengths corresponds to Moore's table
	print 'air wavelength N1 minimum =', wavair[minidx1],'AA, ', wavair[minidx2], 'AA'   

	#wav0 = wavvac[minidx1]*1e-8 # NaI D2 core wavlength in cm
	wav0 = wavair[minidx2]*1e-8 # NaI D1 core wavlength in cm

	P = Part3(wav0)
	#---------------------- plot spectra
	label1=['Na I','Na I']
	P.plot_quantities(wavvac,phspect, plot_label='_no',x_label=r'vacuum wavelength [\AA]', y_label='intensity')
	P.plot_quantities(wavair,phspect, plot_label='_no',x_label=r'air wavelength [\AA]', y_label='intensity')


	#------------------------plot boltzmann distribution

	boltz0 = P.boltz_Na(T,r=1,s=1)
	boltz1 = P.boltz_Na(T,r=1,s=2)
	boltz2 = P.boltz_Na(T,r=1,s=3)

	P.plot_quantities(h,np.array([boltz0,boltz1,boltz2]), plot_label=['$s=1$ ground state', '$s=2$ Na I D1', '$s=3$ Na I D2'], x_label='height [km]',y_label='population fraction $n_{1,s}/N_1$')

	#----------------------- plot saha distribution
	saha0 = np.zeros((len(T),2))
	for k in range(2):
		for i in range(len(T)): 
			saha0[i,k] =  P.saha_Na(T[i],ne[i],k+1)

	P.plot_quantities(h,np.array([saha0[:,0],saha0[:,1]]), plot_label=['Na I ', 'Na II'], x_lim=[-100,2000], y_lim = [1e-4,1e1], x_label='height [km]',y_label=r'ionization state fraction $N_{r}/N_\mathrm{tot}$', logY=True)


	#-------------- NaI D lines
	fudge = np.array([0,1,2]) # 0 is normal gammavdW, 1 is 2*gammavdW, 2 is gammavdW =10**-() from vienna atomic line database

	spectra,wav=calspectra(P,fudge,phspect,wavair, wav0, h, T, vt, ne, nH, nprot, pgas) # different types of van der Waals broadening term gamma.)
	

	#-----------------plot NaI D1 line
	labels = [r'$\gamma_\mathrm{vdW}$',r'$2\gamma_\mathrm{vdW}$',r'$\gamma_\mathrm{vdW}=10^{-7.526}N_H$' ]
	if len(fudge)==1: style=['']
	else: style = ['--','','--']

	P.plot_quantities(wavair[2500:3500],phspect[2500:3500]/max(phspect),plot_label='observed', hold=True)
	P.plot_quantities(wav*1e8,spectra, style, plot_label=labels, x_lim=[5894,5+5893],x_label='wavelength [\AA]', y_label='intensity')

main()