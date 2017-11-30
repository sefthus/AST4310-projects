import matplotlib as mpl
mpl.use('TkAgg') # Ensure that the Tkinter backend is used for generating figures
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import special

class Payne():
	def __init__(self,Ca=False):
		self.chiion=np.array([7,16,31,51])
		if Ca==True:
			#self.chiion=np.array([6,12,51,67])
			self.chiion=np.array([6.113,11.871,50.91,67.15])
		self.kerg = 1.38e-16 # Boltzmann constant erg K
		self.kev =8.61734e-5 # ev/deg
		self.h = 6.62607e-27 # Plack constant
		self.m_e = 9.109390e-28 # electron mass
		self.c = 2.99792e10 # cm*s^-1 speed of light

	def partfunc_E(self,T):
		"""Partition function for Schadeeium """
		chiion, kev= self.chiion,self.kev # ionization energies
		U = np.zeros(len(chiion)) # declare a 4 zero-element array
		for r in range(len(chiion)):
			for s in range(int(chiion[r])):
				U[r] += np.exp(-s /(kev*T))
		return U # returns all the values of u array

	def boltz_E(self,T,r,s):
		"""Boltzmann distribtuion for Schadeeium """

		g = 1.
		kev =self.kev # Boltzmann constant in eV/deg
		U=self.partfunc_E(T)
		n_div_Nr = float(g)/U[r-1]*np.exp(-(s-1)/(kev*T))

		return n_div_Nr

	def saha_E(self,T,el_pressure,ion_stage):
		"""Saha distribtuion for Schadeeium """

		chiion,kerg,kev,h,m_e = self.chiion,self.kerg,self.kev,self.h,self.m_e
		kevT =kev*T
		kergT = kerg*T
		el_density = el_pressure/kergT
		U = self.partfunc_E(T)
		U = np.append(U,2) # add estimated fifth value to get N_4
		sahaconst = 2*(2.*np.pi*m_e*kergT/h**2)**(3./2)/el_density
		Nstage = np.zeros(5)
		Nstage[0]=1.

		for r in range(4):
			Nstage[r+1]=Nstage[r]*sahaconst*U[r+1]/U[r]*np.exp(-chiion[r]/kevT)

		N_total = np.sum(Nstage)
		Nstage_div_Ntot = Nstage/float(N_total)
		if isinstance(ion_stage,(int,float)) ==True: # if only given a single ionization stage
			ion_stage=int(ion_stage)
			return Nstage_div_Ntot[ion_stage-1] # ion stage begins at 1, not 0
		else:	# if given several ionization stages
			return Nstage_div_Ntot

	def sahabolt_E(self,T,el_pressure,r,s):
		""" Saha-Boltzmann distribution for Schadeeium"""

		return self.saha_E(T,el_pressure,r)*self.boltz_E(T,r,s)

	def sahabolt_H(self,T,el_pressure,level):
		""" Saha-Boltzmann distribution for Hydrogen """

		#constants
		kerg,kev,h,m_e =self.kerg,self.kev,self.h,self.m_e
		kevT =kev*T
		kergT = kerg*T
		el_density = el_pressure/kergT

		s_max = 100 # nr of excitation levels s, reasonable high number
		g = np.zeros((2,s_max)) # statistical weights, to many for H
		g[1,0]=1
		chiion_H=13.598 # ionizaion energy eV
		chiexc=np.zeros((2,s_max)) # excitation level energies
		chiexc[1,0]=0.

		for s in range(s_max):
			chiexc[0,s]=chiion_H*(1-1./(s+1)**2)
			g[0,s]=2*(s+1)**2


		# Partition function
		U=np.zeros(2)
		U[0]=np.sum(g[0,:]*np.exp(-chiexc[0,:]/kevT))
		U[1]=g[1,0]

		#Saha
		sahaconst = 2*(2.*np.pi*m_e*kergT/h**2)**(3./2)/el_density
		Nstage = np.zeros(2)
		Nstage[0]=1.
		Nstage[1]=Nstage[0]*sahaconst*U[1]/U[0]*np.exp(-chiion_H/kevT)
		N_total=np.sum(Nstage)
		Nstage_div_Ntot = Nstage/N_total

		#Boltzmann
		n = Nstage[0]*g[0,level-1]/U[0]*np.exp(-chiexc[0,level-1]/(kev*T))

		#Saha-Boltzmann
		return n/N_total


	def plot_formatting(self,fam='serif',fam_font='Computer Modern Roman',font_size=15,tick_size=15):
		""" you get to define what font and size of xlabels and axis ticks you"""
		"""like, if you want bold text or not.								  """
	
		plt.rc('text',usetex=True)
		axis_font={'family': fam,'serif':[fam_font],'size':font_size}
		plt.rc('font',**axis_font)
		plt.rc('font',weight ='bold')
		#plt.rcParams['text.latex.preamble']=[r'\boldmath']
		plt.xticks(fontsize=tick_size)
		plt.yticks(fontsize=tick_size)


def table():
	""" Reproduces the table values for Schadeeium"""
	solve=Payne()
	print 'Schadee\'s first table:'
	print 'T=5000 K:',solve.partfunc_E(T=5000)
	print 'T=10e3 K:', solve.partfunc_E(T=1e4)
	print 'T=20e3 K:',solve.partfunc_E(T=2e4),'\n'


	s = np.linspace(1,15,15)
	print 'Schadee\'s second table:'
	print 'T=5000 K:\n', solve.boltz_E(T=5000,r=1,s=s)
	print 'T=10e3 K:\n', solve.boltz_E(T=10000,r=1,s=s)
	print 'T=20e3 K:\n', solve.boltz_E(T=20000,r=1,s=s),'\n'

	r=np.linspace(1,5,5)
	print 'Schadee\'s third table:'
	print 'T=5000 K:\n', solve.saha_E(T=5000,el_pressure=1e3,ion_stage=r)
	print 'T=10e3 K:\n', solve.saha_E(T=10000,el_pressure=1e3,ion_stage=r)
	print 'T=20e3 K:\n', solve.saha_E(T=20000,el_pressure=1e3,ion_stage=r)

def population():
	""" Computes Payne curves """
	solve = Payne()
	N=301 # nr of values
	T = np.linspace(0,30000,N)	# K, temperature
	el_pressure=131 #dyne cm^-1 electron density

	pop=np.zeros((4,N,4))	#n_r/N populations
	for i in range(1,N):
		for r in range(1,5):
			for s in range(1,5):
				pop[r-1,i-1,s-1]= solve.sahabolt_E(T[i],el_pressure,r,s)


	solve.plot_formatting()
	cmap = mpl.cm.Set1 # color of plots

	[plt.plot(T,pop[i,:,s],color=cmap(s/6.)) for i in range(4) for s in range(4)]
	plt.legend(['$s = 1$','$s = 2$','$s = 3$','$s = 4$'],loc='best')
	plt.yscale('log')
	plt.text(0,1.5,'$r = 1$')
	plt.text(6000,1.2, '$r = 2$')
	plt.text(12500,1., '$r = 3$')
	plt.text(22500,.7, '$r = 4$')
	#plt.title('Ionization stages $r$ and excitation states $s$ for Schadeenium')
	plt.ylim([1e-4,3])
	plt.xlabel('temperature [K]')
	plt.ylabel('population $n_{r,s}/N$')
	plt.show()

def hydrogen():
	solve = Payne()
	solve.sahabolt_H(T=5000,el_pressure=1e2,level=1)

def Ca_line_strength():
	solve = Payne(Ca=True) # To compute for Ca not E
	T = np.arange(1000,20001,100)
	CaH = np.zeros(T.shape)
	Ca_abund = 2e-6

	for i in range(190):

		NH = solve.sahabolt_H(T[i],el_pressure=1e2,level=2)
		NCa = solve.sahabolt_E(T[i],el_pressure=1e2,r=2,s=1)
		CaH[i]=NCa*Ca_abund/NH

	#Find index of the temperature closest to T=5000 K 
	index = np.argwhere(abs(T-5000)<1e-3)[0][0]
	print 'Ca/H at temperature 5000 K:  ',CaH[index]

	solve.plot_formatting()
	plt.plot(T,CaH)
	plt.yscale('log')
	plt.xlabel('temperature [K]')
	plt.ylabel('Ca II K/H$\\alpha$')#'population $n_{r,s}/N$')
	plt.show()

def Ca_temp_sensitivity():
	solve = Payne(Ca=True) # To compute for Ca not E
	T = np.arange(2000,12001,100)
	dNCadT = np.zeros(T.shape)
	dNHdT = np.zeros(T.shape)
	dT = 1.

	for i in range(len(T)):

		NH = solve.sahabolt_H(T[i],el_pressure=1e2,level=2)
		NH2 = solve.sahabolt_H(T[i]-dT,el_pressure=1e2,level=2)
		dNHdT[i] = (NH-NH2)/(dT*NH)

		NCa = solve.sahabolt_E(T[i],el_pressure=1e2,r=2,s=1)
		NCa2 = solve.sahabolt_E(T[i]-dT,el_pressure=1e2,r=2,s=1)
		dNCadT[i] = (NCa-NCa2)/(dT*NCa)

	solve.plot_formatting()
	plt.plot(T,np.fabs(dNCadT),T,np.fabs(dNHdT))
	plt.yscale('log')
	plt.xlabel('temperature [K]')
	plt.ylabel('$\left| \left( \Delta n(r,s) / \Delta T \\right) / n(r,s) \\right|$')#'population $n_{r,s}/N$')
	#plt.show()

	NCa = np.zeros(T.shape)
	NH = np.zeros(T.shape)
	for i in range(len(T)):
		NH[i] = solve.sahabolt_H(T[i],el_pressure=1e2,level=2)
		NCa[i] = solve.sahabolt_E(T[i],el_pressure=1e2,r=2,s=1)

	plt.plot(T,NCa/np.amax(NCa),'--',T,NH/np.amax(NH),ls='--')
	plt.ylim(1e-8,1.3)
	plt.legend(['CaII$(s=1)$','HI$(s=2)$','rel. pop. CaII$(s=1)$','rel. pop. HI$(s=2)$'])
	plt.show()


	"""
	# Checks that the max of the rel. pop. is
	# at the same place as the min of the  T sensitivity curves
	
	index1 = np.argwhere(NCa==np.amax(NCa))[0][0]
	index11 = np.argwhere(np.fabs(dNCadT)==np.amin(np.fabs(dNCadT)))[0][0]

	index2 = np.argwhere(NH==np.amax(NH))[0][0]
	index22 = np.argwhere(np.fabs(dNHdT)==np.amin(np.fabs(dNHdT)))[0][0]
	print T[index1],T[index11]
	print T[index2],T[index22]
	"""
def cool_stars():
	""" Plot and print the population of neutral hydrogen"""
	""" as a function of temperature"""
	solve = Payne()
	for T in np.arange(2e3,1.5e4+1,1e3):
		print T, solve.sahabolt_H(T,1e2,1)

	T = np.linspace(1000,20000,1000)
 	nH = np.zeros(T.shape)
	for i in range(len(T)):
		nH[i]=solve.sahabolt_H(T[i],1e2,1)

	index = np.argwhere(nH<=0.5)[0][0]
	print "Temperature when about 50% of neutral hydrogen is ionized: \n",T[index], ', nH:',nH[index]
	solve.plot_formatting()
	plt.plot(T,nH,T[index],nH[index],'o')
	plt.xlabel('temperature [K]')
	plt.ylabel('neutral hydrogen fraction')

	plt.show()

def strength_ratio():
	T = 5000 # K
	kev =8.61734e-5 # ev/deg
	kevT = kev*T
	g = np.array([2,8,16,32])
	chi = np.array([0,10.2,12.09,12.75])
	lineratio=['Ly alpha/Ba alpha','Ba alpha/Pa alpha','Pa alpha/Br alpha']
	line = np.zeros(len(g))
	for i in range(len(g)-1):
		line[i] = float(g[i])*np.exp(-chi[i]/(kevT))
		line[i+1]=float(g[i+1])*np.exp(-chi[i+1]/(kevT))
		ratio = line[i]/line[i+1]
		print lineratio[i], ratio
	print '\n Ba alpha/Br alpha ', line[1]/line[-1] 
strength_ratio()
#cool_stars()
#Ca_temp_sensitivity()
#Ca_line_strength()
#hydrogen()
#population()
#table()