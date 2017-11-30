import matplotlib as mpl
mpl.use('TkAgg') # Ensure that the Tkinter backend is used for generating figures
import numpy as np 
import matplotlib.pyplot as plt
from scipy import special

class Minnaert():
	def __init__(self,Ca=False):
		self.kerg = 1.38e-16 # Boltzmann constant erg K
		self.kev =8.61734e-5 # ev/deg
		self.h = 6.62607e-27 # Plack constant
		self.c = 2.99792e10 # cm*s^-1 speed of light

	def planck(self,T,wav):
		h,c,kerg=self.h,self.c,self.kerg
		return 2*h*c**2/wav**5/(np.exp(h*c/(wav*kerg*T))-1) 

	def voigt(self,gamma,x):
		z = x+1j*gamma
		V = special.wofz(z).real
		return V

	def profile(self,a,tau0,u,T_s=5700,T_l=4200):
		wav = 5000e-8
		intensity = np.zeros(u.size)
		usize = u.size
		for i in range(usize):
			tau = tau0*self.voigt(a,u[i])
			intensity[i] = self.planck(T_s,wav)*np.exp(-tau) + self.planck(T_l,wav)*(1.-np.exp(-tau))

		return intensity

	def plot_formatting(self,fam='serif',fam_font='Computer Modern Roman',font_size=14,tick_size=14):
		""" you get to define what font and size of xlabels and axis ticks you"""
		"""like, if you want bold text or not.								  """
	
		plt.rc('text',usetex=True)
		axis_font={'family': fam,'serif':[fam_font],'size':font_size}
		plt.rc('font',**axis_font)
		plt.rc('font',weight ='bold')
		#plt.rcParams['text.latex.preamble']=[r'\boldmath']
		plt.xticks(fontsize=tick_size)
		plt.yticks(fontsize=tick_size)

def equivelent_width(one=False):
	""" Computes the equivalent width of spectral lines and the curve of growth"""
	solve = Minnaert()

	T_s = 5700
	T_l = 4200	#absorption
	T_l = 6000	# emission
	u = np.arange(-200,200.4,0.4)
	n = abs((u[-1]-u[0]))/len(u)
	print n
	a=0.1	# dampening
	# Compute only one tau0 value
	if one==True: tau0=np.array([1e2])
	else: tau0 = np.logspace(-2,4,61)


	eqw = np.zeros(tau0.shape)
	
	for i in range(len(tau0)):
		intensity= solve.profile(a,tau0[i],u,T_s,T_l)
		rel_depth = (intensity[0]-intensity)/intensity[0]
		eqw[i] = np.sum(rel_depth)*n

	solve.plot_formatting()
	if one==True: 
		print eqw
		plt.plot(u,rel_depth)
		plt.ylabel('relative depth $(I_\lambda(0)-I_\lambda)/I_\lambda(0)$')
		plt.xlabel('$u$')
	else:
		plt.plot(tau0,np.fabs(eqw))
		plt.xlabel('$\\tau(0)$')
		if T_s<T_l:
			plt.ylabel('Abs. value of Equivalent width $W_\lambda$')
		else:plt.ylabel('Equivalent width $W_\lambda$')
		plt.yscale('log'); plt.xscale('log')
	plt.show()


def emergent_line_profiles(relative=True,one=False):
	""" Computes Schuster-Schwarzchild line profiles. u vs Intensity"""
	solve = Minnaert()

	T_s = 5700	# solar surface temperature
	T_l = 4200	# solar T_min temperature, absoprtion
	T_l=6000	#emission
	a = 0.1		# dampening paramter
	if one==True:	#Calculate only for one wavelength and tau0
		wav = np.array([5000e-8]) # wavelength in cm
		tau0=np.array([1])		# reversing layer thickness
	else:
		wav = np.array([2e3,5e3,10e3])*1e-8 # wavelength in cm
		tau0=np.array([0.01,0.05,0.1,0.5,1,5,10,50])
	u = np.arange(-10,10.1,0.1)	# wavelength seperation, dim.less
	intensity = np.zeros(u.shape)# to be filled, cm^-3 s^-1 sr^-1
	solve.plot_formatting()


	if relative==True: 
		#legends=['a','_nolegend_','_nolegend_']
		tau0=np.array([0.01,1,50])
		colors = ['g','b','r']
		for k in range(len(wav)):
			legends=['$\lambda=$'+np.str(int(wav[k]*1e8))+'\AA','_nolegend_','_nolegend_']
			B = solve.planck(T_l,wav[k])
			for n in range(len(tau0)):
				for i in range(201):
					tau = tau0[n]*solve.voigt(a,u[i])
					intensity[i]=solve.planck(T_s,wav[k])*np.exp(-tau)+solve.planck(T_l,wav[k])*(1.-np.exp(-tau))
				intensity=intensity/np.amax(intensity)
				plt.plot(u,intensity,label=legends[n],color=colors[k])

		plt.legend()
		plt.ylabel('Relative intensity $I_\lambda/I_\lambda(0)$')
		plt.yscale('log')
		plt.xlabel('$u$')
		plt.title('$\\tau(0)=[0.01,\,1,\,50]$')
		plt.tight_layout()
		plt.show()
	else:
		for k in range(len(wav)):
			B = solve.planck(T_l,wav[k])
			for n in range(len(tau0)):
				for i in range(201):
					tau = tau0[n]*solve.voigt(a,u[i])

					intensity[i]=solve.planck(T_s,wav[k])*np.exp(-tau)+solve.planck(T_l,wav[k])*(1.-np.exp(-tau))
			#print np.max(intensity)
				plt.plot(u,intensity,label='$\\tau(0)=\,$'+np.str(tau0[n]))

			plt.axhline(B,linestyle='--')
			#absoprtion
			#plt.annotate('$B_\lambda(T_{layer}=$ '+np.str(T_l)+' K)', xy=(0.65, 0.05), xycoords='axes fraction')
			#plt.annotate('$\lambda = $'+np.str(int(wav[k]*1e8))+' \AA', xy=(0.75, 0.12), xycoords='axes fraction')
			#Emission
			plt.annotate('$B_\lambda(T_{layer}=$ '+np.str(T_l)+' K)', xy=(0.05, 0.90), xycoords='axes fraction')
			plt.annotate('$\lambda = $'+np.str(int(wav[k]*1e8))+' \AA', xy=(0.05, 0.83), xycoords='axes fraction')

			plt.legend()
			plt.ylabel('Intensity $I_\lambda$')
			plt.yscale('log')
			plt.xlabel('$u$')
			plt.tight_layout()
			plt.show()
	
def voigt_profile(ylog=False):
	""" Calculates and plots voigt profiles for different u and a"""
	solve = Minnaert()

	u = np.arange(-10,10.1,0.1)		# dimensionless wavelength seperation
	a = np.array([0.001,0.01,0.1,1])	# dampening
	vau = np.zeros((a.shape[0],u.shape[0]))	# cm, to keep voigt profiles

	solve.plot_formatting()
	for i in range(len(a)):
		vau[i]=solve.voigt(a[i],u)

		plt.plot(u,vau[i,:],label='$a = \,$'+np.str(a[i])) 
	plt.legend()
	if ylog==True:	plt.yscale('log')
	plt.ylabel('Voight profile')
	plt.xlabel('$u$')
	plt.show()

def emergent_intensity(logx=False,logy=False):
	""" Radiation through isothermal layer. Calculates the emergent intensity
		from an isothermal layer, includes the weakened initial intensity, 
		and the intensity that originates in the layer			"""
	solve = Minnaert()

	B=2. # Black body radiation intensity
	tau = np.arange(0.01,10.01,0.01)	# optical thickness/thines
	I0 = np.arange(4,-1,-1)		# Initial beam intensity,cm^-3 s^-1 sr^-1

	solve.plot_formatting()

	# Calculates the emergent intensity and plots against tau 
	for i in range(len(I0)):
		intensity = I0[i]*np.exp(-tau) + B*(1-np.exp(-tau))
		plt.plot(tau,intensity,label='$I_\lambda(0)$ = '+str(I0[i]))

	plt.xlabel('Optical depth $\\tau$')
	plt.ylabel('Intensity $I_\lambda$ [erg cm$^{-3}$ s$^{-1}$ sr$^{-1}$]')
	if logxy==True:
		plt.yscale('log')
		plt.xscale('log')
	#plt.title('Radiation through isothermal layer')
	plt.legend()
	plt.show()

def planck_curves(ylog=False,xlog=False):
	""" Calculates the black body radiation for different temperatures and
		wavelengths, plots the Planck curves against wavelengths """

	solve = Minnaert()
	# Test of the planck function
	print 'BB radiation at T=5000 K and lambda=5000 Angstrom'
	print solve.planck(T=5000,wav=5000e-8)

	T = np.arange(5e3,8e3+1,500)	# temperatures in K
	wav = np.arange(1e3,20801,200) 	# wavelengths in AA

	# Calculates the intensity in cgs,cm^-3 s^-1 sr^-1
	b = np.array([solve.planck(temp,wav*1e-8) for temp in T])

	# plotting Planck curves with/without logarithmic scales
	solve.plot_formatting()
	[plt.plot(wav,b[i]) for i in range(len(b))]
	if ylog == True: plt.yscale('log')	# logarithmic scaling
	if xlog == True: plt.xscale('log')
	plt.legend(['$T = $%d K'%temp for temp in T])
	#plt.title('Black-body radiation')
	plt.xlabel('Wavelength [\AA]')
	plt.ylabel('Radiation intensity [erg cm$^{-3}$ s$^{-1}$ sr$^{-1}$]')
	plt.show()



#planck_curves(ylog=False,xlog=False)
#emergent_intensity(logxy=True)
#voigt_profile(ylog=False)
emergent_line_profiles(relative=False)
#equivelent_width(one=False)