import matplotlib as mpl
mpl.use('TkAgg') # Ensure that the Tkinter backend is used for generating figures
import numpy as np
import matplotlib.pyplot as plt

class Part1():
	def __init__(self):
		self.mH = 1.67352e-24 #g
		self.mprot = 1.67262e-24 #g
		self.me = 9.10939e-28 #g
		self.mHe = 3.97*self.mH
		self.kB = 1.380658e-16 # erg K^-1 boltzmann constant
		self.readfile1 = np.genfromtxt('falc.dat')
		readfile1 = self.readfile1
		self.readfile2 = np.genfromtxt('earth.dat')
		readfile2=self.readfile2
		# FREE protons and e density!!
		self.h,self.tau500,self.colm,self.T,self.vt,self.nH,self.nprot,self.ne,self.Ptot, self.PgasPtot, self.rho = \
						 ( readfile1[:,0], readfile1[:,1], readfile1[:,2],
						readfile1[:,3], readfile1[:,4], readfile1[:,5], readfile1[:,6],
						readfile1[:,7], readfile1[:,8], readfile1[:,9], readfile1[:,10])

		self.hE,self.logPE,self.TE,self.logrhoE,self.lognE = ( readfile2[:,0], readfile2[:,1], readfile2[:,2],
						readfile2[:,3], readfile2[:,4])

		self.nHe = self.nH*0.1
		self.ne_minH = self.ne-self.nprot # + Z/2.*nMet

		nHe, ne_minH = self.nHe, self.ne_minH
		self.Teff = 5770
		self.nphot_lowlevel = 20*self.T**3
		self.nphot_highlevel = 20*self.Teff**3/(2*np.pi)

		h,mH,mprot,me,mHe,kB,h,tau500,colm,T,vt,nH,nprot,ne,Ptot,PgasPtot,rho = \
			(self.h,self.mH,self.mprot,self.me,self.mHe,self.kB,self.h,self.tau500,self.colm,
			self.T,self.vt,self.nH,self.nprot,self.ne,self.Ptot,self.PgasPtot,self.rho)
		 
		
	def plot_quantities(self,x,y,style='',plot_label=' ',x_label='',y_label='',logAxes=False,logX=False,logY=False,hold=False):
		plt.rc('text',usetex=True)
		plt.rc('font',**{'family':'serif','size':14})
		plt.plot(x,y,style,label=plot_label)
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
		plt.show()

	def compute_HpS(self):
		Hp_array = np.linspace(120,160,1000)
		rho_diff = [np.sum(np.abs(self.rho[-1]*np.exp(-self.h[:65]/Hp_array[i])-self.rho[:65])) for i in range(len(Hp_array))]
		idx=np.argmin(np.abs(rho_diff))
		self.Hp=Hp_array[idx]
		rho_Hp = self.rho[-1]*np.exp(-self.h/self.Hp)
		return self.Hp


	def compute_falc(self):
		nHe, ne_minH = self.nHe, self.ne_minH

		h,mH,mprot,me,mHe,kB,h,tau500,colm,T,vt,nH,nprot,ne,Ptot,PgasPtot,rho = \
			(self.h,self.mH,self.mprot,self.me,self.mHe,self.kB,self.h,self.tau500,self.colm,
			self.T,self.vt,self.nH,self.nprot,self.ne,self.Ptot,self.PgasPtot,self.rho)
		
		#----------------plot h vs T
		#self.plot_quantities(h, T, x_label='height [km]', y_label='temperature [K]')
		# plot pTot vs colm
		# ------------------- on subplot form:
		'''
		plt.subplot(2,1,1)
		self.plot_quantities(colm,Ptot, y_label=r'$P_\mathrm{tot}$ [dyn cm$^{-2}$]', x_label='',hold=True)
		plt.subplot(2,1,2)
		self.plot_quantities(colm,Ptot, y_label=r'$P_\mathrm{tot}$ [dyn cm$^{-2}$]', x_label='column mass [g cm$^{-2}$]',logAxes=True)		
		'''
		# ------------------------ calculate g from P=g*m, gravitational something
		gSurface = np.sum(Ptot/colm)/len(Ptot) 
		print 'gSurface = ',gSurface
		# -------------------- calculate some mass densities
		rhoH = nH*mH
		rhoHe = nHe*mHe
		rhoe = ne*me
		rhoprot = nprot*mprot
		rhoe_minH = rhoe-rhoprot # electrons not from hydrogen ionization
	
		# ------------------------- calculate some mass density ratios
		HmassRatio = rhoH/rho#(nH*mH + nprot*mprot + ne*me)
		HemassRatio = rhoHe/rho
		HHemassRatio = HmassRatio + HemassRatio
		MetalsmassRatio = 1- HHemassRatio
		meanMetRatio = np.sum(MetalsmassRatio)/len(MetalsmassRatio)
		print '\nmax and min H ratio:   ', np.amax(HmassRatio), np.amin(HmassRatio), 'difference %: ', 100*(np.amax(HmassRatio)-np.amin(HmassRatio))
		print 'max and min H+He ratio:', np.amax(HHemassRatio), np.amin(HHemassRatio), 'difference %: ', 100*(np.amax(HHemassRatio)-np.amin(HHemassRatio))
		print 'Mean Metal fraction = ', meanMetRatio
		plt.rc('text',usetex=True)
		plt.rc('font',**{'family':'serif','size':14})

		# ---------------------- plot h vs H and He mass ratios
		#ax1=plt.subplot(2,1,1)
		#self.plot_quantities(h,HmassRatio,y_label=r'$\rho_H/\rho$',hold=True)
		#plt.setp(ax1.get_xticklabels(), visible=False)
		#ax2=plt.subplot(2,1,2,sharex = ax1)
		#self.plot_quantities(h,HHemassRatio,y_label=r'$\rho_{H+He}/\rho$', x_label='height [km]',hold=True)
		#plt.subplots_adjust(hspace=.1)
		#plt.show()

		# ---------------------column mass against height
		#ax1=plt.subplot(2,1,1)
		#self.plot_quantities(h,colm,  y_label='column mass [g cm$^{-2}$]',hold=True)
		#plt.setp(ax1.get_xticklabels(), visible=False)
		#ax2=plt.subplot(2,1,2, sharex=ax1)
		#self.plot_quantities(h,colm,  y_label='column mass [g cm$^{-2}$]',x_label='height [km]',logY=True,hold=True)
		#plt.subplots_adjust(hspace=.1)
		#plt.show()

		# calculate scale height Hp deep in the photosphere, least square
		#--------------------------- Note Hp is actually Hrho, just easier to write Hp
		Hp=self.compute_HpS()


		print '\nThe density scale height Hp =',Hp

		# --------------------------- plot h vs density and density from Hp
		#self.plot_quantities(h,rho,  plot_label='Observed FALC density', hold=True)#,logY=True)
		#self.plot_quantities(h,rho_Hp, plot_label='Analytical density', y_label='gas density [g cm$^{-3}$]',x_label='height [km]')

		#--------------computing gas pressure
		Pgas = PgasPtot*Ptot

		#--------------------------------plot gas pressure against height
		#self.plot_quantities(h,Pgas,'c',plot_label='$P_{gas}$', hold = True)
		#self.plot_quantities(h,(nH+ne)*kB*T,'r',plot_label='$(n_H+n_e)k_BT$', hold = True)
		#self.plot_quantities(h,(nH+ne+nHe)*kB*T,'k--',plot_label='$(n_H+n_e + n_{He})k_BT$', y_label='gas pressure [dyn cm$^{-2}$]', x_label='height [km]')

		#--------------------------- plot ideal gas pressure to FALC pressure ratio
		#self.plot_quantities(h,(nH+ne)*kB*T/Pgas, plot_label='$(n_H+n_e)k_BT/P_{gas}$', hold = True)
		#self.plot_quantities(h,(nH+ne+nHe)*kB*T/Pgas, plot_label='$(n_H+n_e + n_{He})k_BT/P_{gas}$',y_label='pressure ratio $nk_B T/P_{gas}$ ',x_label='height [km]')

		# -------------------plot Hydrogen, electron, proton density vs height
		self.plot_quantities(h,(nH-nprot),plot_label=r'$n_H$', hold=True)
		self.plot_quantities(h,ne, plot_label=r'$n_e$', hold=True)
		self.plot_quantities(h,nprot,plot_label=r'$n_p$',hold = True)
		self.plot_quantities(h,ne_minH, plot_label=r'$n_{e,He+Z}$',x_label='height [km]', y_label='number density [cm$^{-3}$]',logY=True)

		# -------------------plot hydrogen ionization fraction
		self.plot_quantities(h,nprot/nH,x_label='height [km]', y_label=r'H$^+$ to H fraction   $\,\,n_p/n_H$',logY=True)

		# -----------------calculate photon density
		#Teff = 5770
		nphot_lowlevel = 20*T[-1]**3
		#self.nphot_highlevel = 20*Teff**3/(2*np.pi)
		nphot_highlevel = self.nphot_highlevel	# high in the atmosphere
		nphot_lowlevel = self.nphot_lowlevel[-1] # low in the atmosphere

		#----------------------------- print some values
		print '\nAt h =',h[-1], ', tau=:',tau500[-1],':'
		print '  n_photon   = ', nphot_lowlevel,'\n  n_hydrogen = ', nH[-1]
		print '  ratio = ',nphot_lowlevel/nprot[-1]
		print 'At h =',h[0], ', tau=',tau500[0],':'
		print '  n_photon   = ', nphot_highlevel,'\n  n_hydrogen =  %e'% nH[0]
		print '  ratio = ', nphot_highlevel/nprot[0]

	def compute_earth(self):
		#--------------------- for Earth data
		kB,mH, hE, logPE, TE, logrhoE, lognE = self.kB, self.mH, self.hE, self.logPE, self.TE, self.logrhoE, self.lognE
		TS, PS, rhoS, ne, nHe, nH, nprot = self.T, self.Ptot, self.rho, self.ne, self.nHe, self.nH, self.nprot
		nphot_highlevel, nphot_lowlevel = self.nphot_highlevel, self.nphot_lowlevel

		#-------------------------- some constants
		R = 6.957E10 	# solar radius in cm
		D = 1.496e13	# cm distance earth sun
		gE = 980.665 	# cm s^-2
		nS = ne + nHe + nH + nprot
		muS = rhoS/(mH*nS)
		PE = 10**logPE
		nE = 10**lognE
		rhoE = 10**logrhoE
		muE= rhoE/(mH*nE)	# mean molecular weight
		#idx = np.argmin(np.abs(rhoE - rhoE[0]/np.e))
		#--------------------------- Note Hp is actually Hrho, just easier to write Hp
		HpE =  kB*TE[0]/(muE[0]*mH*gE)/1e5 # in km. Correct! 
		HpS = self.compute_HpS()
		rho_HpE = rhoE[0]*np.exp(-hE/HpE)
		print 'density scale height Earth HpE = ', HpE,'km'
		print 'density scale height Sun HpS   = ', HpS, 'km'
		nphot = np.pi*(R/D)**2*nphot_highlevel	# photon density in Earh atmosphere

		# ------------------- plot temperature, pressure, density and number density against height
		'''
		self.plot_quantities(hE,TE, x_label='height [km]', y_label='temperature [K]')
		ax1=plt.subplot(3,1,1)
		self.plot_quantities(hE,PE, y_label='$P$ [dyn cm$^{-2}$]',hold=True,logY=True)
		plt.setp(ax1.get_xticklabels(), visible=False)
		ax2=plt.subplot(3,1,2,sharex=ax1)
		self.plot_quantities(hE,rhoE,  y_label='$\\rho$ [g cm$^{-3}$]',hold=True,logY=True)
		plt.setp(ax2.get_xticklabels(), visible=False)
		ax3=plt.subplot(3,1,3,sharex = ax2)
		self.plot_quantities(hE,nE,  x_label='height [km]', y_label='$n_E$ [cm$^{-3}$]',hold=True,logY=True)
		plt.subplots_adjust(hspace=.1)
		plt.show()
		'''
		# density and pressure together normalized units


		idx2 = [np.argwhere(PE/np.amax(PE)<rhoE/np.amax(rhoE))][0]
		#print hE[idx2[-1]]
		
		# ------------------------- plot normalized pressure and density together against height
		#self.plot_quantities(hE,TE/np.amax(TE),style='r--',plot_label='$T/T_{max}$', hold=True)
		self.plot_quantities(hE,PE/np.amax(PE), plot_label='$P/P_{max}$',hold=True)
		self.plot_quantities(hE,rhoE/np.amax(rhoE), plot_label=r'$\rho/\rho_{max}$', x_label='height [km]',y_label=r'pressure, gas density', hold=False)#,logY=True)

		#--------------------------plot mean molecular weight vs height
		#self.plot_quantities(hE,muE, x_label='height [km]', y_label= 'mean molecular weight')

		#--------------------------plot density and density from Hp together against height
		#self.plot_quantities(hE,rhoE,plot_label='Observed density',hold=True)
		#self.plot_quantities(hE,rho_HpE,style='--', plot_label='Analyical density', x_label='height [km]',y_label='gas density [g cm$^{-3}$]')

		idx = np.argmin(np.abs(self.h))
		# ----------------- compare Sun and Earth parameters
		P_ratio = PE[0]/PS[idx] 
		n_ratio = nE[0]/nS[idx] 
		rho_ratio = rhoE[0]/rhoS[idx] 
		T_ratio = TE[0]/TS[idx]
		mu_ratio = muE[0]/muS[idx] 
		print '\nAt h=0:\n  particle density ratio nE/nS     = ', n_ratio
		print '  pressure ratio PE/PS             = ', P_ratio
		print '  density ratio rhoE/rhoS          = ', rho_ratio
		print '  temperature ratio TE/TS          = ', T_ratio
		print '  mean mol. weight ratio muE/muS   = ', mu_ratio

		# ------------------calculate column mass maybe change to -1
		colmE = PE/gE # g cm^-2 column mass, P=F/A = Mg/A
		print '\ncolumn mass at hE = 0 colmE = ', colmE[0]
		print 'column mass at hS = 0 colmS = ', self.colm[idx]
		print 'their ratio colmE/colmS = ', colmE[0]/self.colm[idx]
		# --------------------------photon densities, earth sun
		print '\nNphot     =  %e cm^-3'% nphot 
		print 'Nphot_low = ', nphot_lowlevel[idx], 'cm^-3'
		print 'nE(h=0)   =  %e cm^-3' % nE[0] 

		# --------------------- plot numberdensity of particles vs height.
		self.plot_quantities(hE,nE, plot_label='$n_E^{gas}$',x_label='height [km]', y_label='number density [cm$^{-3}$]',hold=True,logY=True)
		plt.axhline(nphot_lowlevel[idx],c='r',ls='--',label='$n_{phot}^{\odot}(0)$')
		plt.axhline(nphot,c='k',ls='--',label='$n_{phot}^{E}$')
		plt.legend()
		plt.show()
		
if __name__=='__main__':
	P = Part1()
	P.compute_falc()
	#P.compute_earth()