import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from readfalc import Part1

class Part2():
	def __init__(self):
		self.kerg = 1.38e-16 # Boltzmann constant erg K
		self.kev =8.61734e-5 # ev/deg
		self.hp = 6.62607e-27 # erg s Plack constant
		self.ccm = 2.99792e10 # cm*s^-1 speed of light
		self.cmum = self.ccm*1e4 # mum*s^-1 speed of light
		self.readfile1 = np.genfromtxt('solspect.dat')
		self.wav,self.Fsmooth, self.Fcont, self.Ismooth,self.Icont = np.loadtxt('solspect.dat',usecols=(0,1,2,3,4), unpack=True)
		self.h,self.tau500,self.colm,self.T,self.vt,self.nH,self.nprot,self.ne,self.Ptot, self.PgasPtot, self.rho = np.loadtxt("falc.dat",usecols=(0,1,2,3,4,5,6,7,8,9,10), unpack=True)
		

	def plot_quantities(self, x,y,style='',plot_label='_nolabel',x_label='',y_label='',logAxes=False,logX=False,logY=False,hold=False):
		plt.rc('text',usetex=True)
		plt.rc('font',**{'family':'serif','size':13})
		if y.ndim> 1: # if x is a array of arrays
			if plot_label=='_nolabel': [plt.plot(x, y[i],style) for i in range(len(y[:,0]))]
			else:	[plt.plot(x, y[i], style,label=plot_label[i]) for i in range(len(y[:,0]))]
			#if planet_label !='_nolegend_': plt.legend()
		else: 	plt.plot(x,y,style,label=plot_label)
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

	def planck(self,T,wav):
		hp,c,kerg=self.hp,self.ccm,self.kerg
		#return 2*hp*wav**3/c**2/(np.exp(hp*wav/(kerg*T))-1.)
		return 2*hp*c**2/wav**5/(np.exp(hp*c/(wav*kerg*T))-1) 

	def T_brightness(self,I,wav):
		hp,c,k=self.hp,self.ccm,self.kerg
		if len(I) !=len(wav):
			Tb = np.zeros((len(wav),len(I)))
			for i in range(len(wav)):
				Tb[i]=hp*c/(wav[i]*k)/np.log(2*hp*c**2/(I*wav[i]**5)+1)
			return Tb
		else:	return hp*c/(wav*k)/np.log(2*hp*c**2/(I*wav**5)+1)

	def compute_LMS(self,T,I,wav,freq):
		BBcurve=np.array([self.planck(T[i],wav*1e-4) for i in range(len(T))])*wav*1e-4/freq
		I_diff = np.sum(np.abs(BBcurve-I),axis=1)
		idx=np.argmin(np.abs(I_diff))
		T_val=T[idx]
		return T_val, BBcurve[idx,:]


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
		k,hp,c=self.kerg,self.hp,self.ccm#1.380658e-16 # Boltzmann constant [erg/K]

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
		kappabf=kappabf*(1.-np.exp(-hp*c/(wav*1E-8*k*temp))) # correct stimulated
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

	def emergint(self,tau5,temp,wl,mu=1.):
		P = Part1()
		nel,nhyd,nprot,h = self.ne, self.nH, self.nprot, self.h
		sigma_Thomson = 6.648e-25 # Thomson cross-section [cm^2]

		wl = wl # wavelength in micron, 1 micron = 1e-6 m = 1e-4 cm = 1e4 Angstrom
		ext = np.zeros(len(tau5))
		tau = np.zeros(len(tau5))
		integrand = np.zeros(len(tau5))
		contfunc = np.zeros(len(tau5))
		intt = 0.0
		hint = 0.0
		for i in range(1, len(tau5)):
			ext[i] = (self.exthmin(wl*1e4, temp[i], nel[i])*(nhyd[i]-nprot[i]) + sigma_Thomson*nel[i])
			tau[i] = tau[i-1] + 0.5 * (ext[i] + ext[i-1]) * (h[i-1]-h[i])*1E5
			integrand[i] = self.planck(temp[i],wl*1e-4)*np.exp(-tau[i]/mu)
			intt += 0.5*(integrand[i]+integrand[i-1])*(tau[i]-tau[i-1])/mu
			hint += h[i]*0.5*(integrand[i]+integrand[i-1])*(tau[i]-tau[i-1])
			contfunc[i] = integrand[i]*ext[i]
		
		# note : exthmin has wavelength in [Angstrom], planck in [cm]
		hmean = hint / intt

		return contfunc, intt, hmean, tau, wl

		#print 'computed continuum intensity wl =%g : %g erg s-1 cm-2 ster-1 cm-1'%(wl, intt)
		#w = np.where(wav == wl)
		#print 'observed continuum intensity wav=%g : %g erg s-1 cm-2 ster-1 cm-1' %(wav[w], Icont[w]*1e10*1e4)

	def contribution_wav(self,wavlist,wav,h,tau5,Tsun, Icont,plot_cont=False,plot_hmean=False,hold=False,legends=True):
		plt.rc('text',usetex=True)
		plt.rc('font',**{'family':'serif','size':14})
		tau_idx = np.zeros(len(wavlist))
		hmean_arr = np.zeros(len(wavlist))
		colors = iter(cm.Set1(np.linspace(0, 1, len(wavlist))))
		contfunc_arr = np.zeros((len(wavlist),len(h)))
		tau_arr = np.zeros((len(wavlist),len(h)))
		hmean_arr = np.zeros(len(wavlist))
		intt_arr = np.zeros(len(wavlist))
		for i in range(len(wavlist)):
			contfunc_arr[i], intt_arr[i], hmean, tau_arr[i], wl=self.emergint(tau5,Tsun,wavlist[i])
			print 'computed continuum intensity wav =%g : %g erg s-1 cm-2 ster-1 cm-1'%(wl, intt_arr[i])
			w = np.where(wav == wl)
			print 'observed continuum intensity wav =%g : %g erg s-1 cm-2 ster-1 cm-1' %(wav[w], Icont[w]*1e10*1e4)
			print 'ratio Iobs/comp =', (Icont[w]*1e10*1e4)[0]/intt_arr[i]
			tau_idx[i] = np.argmin(np.abs(tau_arr[i]-1))
			hmean_arr[i] = hmean
			#print tau[int(tau_arr[i])]
		if plot_cont == True:
			[plt.plot(h,contfunc_arr[i]/np.amax(contfunc_arr[i]), label=r'$\lambda = $ '+str(wavlist[i])) for i in range(len(wavlist))]
			if plot_hmean == True:
				[plt.axvline(hmean_arr[i],ls='--',c=next(colors), label=r'$\langle h(\lambda=$'+str(wavlist[i])+r')$\rangle$') for i in range(len(wavlist))]

			plt.xlabel('height [km]')
			plt.ylabel('contribution function')
			if hold==False:
				if legends==True: plt.legend()
				plt.show()

		return tau_arr,tau_idx.astype(int),hmean_arr,intt_arr

	def emergint_wav(self,wavlist,tau5,Tsun, mu=1):
		if np.isscalar(mu)==True:
			return np.array([self.emergint(tau5,Tsun,wavlist[i],mu=mu)[1] for i in range(len(wavlist))])
		else:
			intt_arr = np.zeros((len(wavlist),len(mu)))
			for i in range(len(wavlist)):
				for k in range(len(mu)):
					intt_arr[i,k]=self.emergint(tau5,Tsun,wavlist[i],mu=mu[k])[1]
	
			return intt_arr
	def flux_int(self,wav,Fcont,tau5):	
		# SSB 2.7 page 17: flux integration
		# ===== three-point Gaussian integration intensity -> flux
		# abscissae + weights n=3 Abramowitz & Stegun page 916
		P = Part1()
		sigma_Thomson = 6.648e-25 # cm^2 Thomson scattering
		h,nhyd,nprot,nel,temp= self.h, self.nH, self.nprot, self.ne, self.T
		xgauss=[-0.7745966692,0.0000000000,0.7745966692]
		wgauss=[ 0.5555555555,0.8888888888,0.5555555555]
		fluxspec = np.zeros(len(wav),dtype=float)
		intmu = np.zeros((3,len(wav)), dtype=float)
		for imu in range(3):
			mu=0.5+xgauss[imu]/2. # rescale xrange [-1,+1] to [0,1]
			wg=wgauss[imu]/2. # weights add up to 2 on [-1,+1]
			for iw in range(0,len(wav)):
				wl=wav[iw]
				ext = np.zeros(len(tau5))
				tau = np.zeros(len(tau5))
				integrand = np.zeros(len(tau5))
				intt = 0.0
				for i in range(1, len(tau5)):
					ext[i] = (self.exthmin(wl*1e4, temp[i], nel[i])*(nhyd[i]-nprot[i]) + sigma_Thomson*nel[i])
					tau[i] = (tau[i-1] + 0.5 * (ext[i] + ext[i-1]) *
					(h[i-1]-h[i])*1E5)
					integrand[i] = self.planck(temp[i],wl*1e-4)*np.exp(-tau[i]/mu)
					intt += 0.5*(integrand[i]+integrand[i-1])*(tau[i]-tau[i-1])/mu
				intmu[imu,iw]=intt
				fluxspec[iw]=fluxspec[iw] + wg*intmu[imu,iw]*mu
		fluxspec *= 2 # no np.pi, Allen 1978 has flux F, not {\cal F}
		figname='part2_fluxintegration'
		f=plt.figure(figname)
		plt.rc('text',usetex=True)
		plt.rc('font',**{'family':'serif','size':13})
		plt.plot(wav,Fcont, label='observed $F^c_\lambda$')
		plt.plot(wav,fluxspec*1e-14, label='FALC $F^c_\lambda$')
		plt.legend(loc='upper right')
		#plt.title('observed and computed continuum flux')
		#plt.ylabel(r'astrophysical flux [$10^{14}$ erg s$^{-1}$ cm$^{-2}$ ster$^{-1}$ cm$^{-1}$]')
		plt.ylabel(r'astrophys. flux 1E14 [erg s$^{-1}$ cm$^{-2}$ ster$^{-1}$ cm$^{-1}$]')
		plt.xlabel('wavelength [$\mu$m]')
		#plt.grid(True)
		plt.tight_layout()
		plt.show()
		f.savefig(figname+'.pdf',bbox_inches='tight')
		#f.savefig(figname+'.png',bbox_inches='tight')

def main():
	P=Part2()
	cmum=P.cmum
	wav,Fsmooth, Fcont, Ismooth,Icont = P.wav,P.Fsmooth, P.Fcont, P.Ismooth, P.Icont
	deltanu = 1. # spectral bendwidth Hz
	# convert to per bandwidth instead of per wav
	freq = cmum/wav
	Icontnu = Icont*wav/freq * 1e10
	Ismoothnu = Ismooth*wav/freq * 1e10
	Fcontnu = Fcont*wav/freq * 1e10
	Fsmoothnu = Fsmooth*wav/freq * 1e10
		
	print 'max(Ic_lambda) =', np.max(Icont), '1E10 erg cm^-2 s^-1 ster^-1 mum^-1, at wav=', wav[np.argmax(Icont)], 'mu m'
	labels1 =[r'$F_\lambda$', r'$F_\lambda^c$', r'$I_\lambda$', r'$I_\lambda^c$']
	#P.plot_quantities(wav,np.array([Fsmooth,Fcont,Ismooth,Icont]), plot_label=labels1, x_label=r'wavelength [$\mu$m]', y_label= r'intensity, a. flux 1E10 [erg cm$^{-2}$s$^{-1}$$\mu$m$^{-1}$ster$^{-1}$]' )

	labels2 =[r'$F_\nu$', r'$F_\nu^c$', r'$I_\nu$', r'$I_\nu^c$']
	print 'max(Ic_nu) =', np.max(Icontnu), ',erg cm^-2 s^-1 ster^-1 Hz^-1, at wav=', wav[np.argmax(Icontnu)], 'mu m'
	#P.plot_quantities(wav,np.array([Fsmoothnu,Fcontnu,Ismoothnu,Icontnu])/1e-5, plot_label=labels2, x_label=r'wavelength [$\mu$m]', y_label= r'intensity, a. flux 1E-5 [erg cm$^{-2}$s$^{-1}$Hz$^{-1}$ster$^{-1}$]' )
	
	T_values = np.linspace(2e3,8e3,10000)
	T,BB_curve=P.compute_LMS(T_values,Icontnu,wav,freq)
	print 'T for Planck curve fitting T=',T, 'K'
	labels3= [r'$I_\nu^c$', r'$B_\nu^c$']
	labels3= [r'$I_\lambda^c$', r'$B_\lambda^c$']
	#P.plot_quantities(wav,np.array([Icontnu,BB_curve])/1e-5, plot_label=labels3, x_label=r'wavelength [$\mu$m]', y_label= r'intensity 1E-5 [erg cm$^{-2}$s$^{-1}$Hz$^{-1}$ster$^{-1}$]')
	#P.plot_quantities(wav,np.array([Icont,BB_curve*freq/wav/1e10]), plot_label=labels3, x_label=r'wavelength [$\mu$m]', y_label= r'intensity 1E10 [erg cm$^{-2}$s$^{-1}\mu$m$^{-1}$ster$^{-1}$]',hold=True)

	Tb = P.T_brightness(Icont*1e10*1e4,wav/1e4)
	#P.plot_quantities(wav,Tb*4/np.max(Tb),x_label=r'wavelength [$\mu$m]',y_label='brightness temperature [K]')
	a=np.argmax(Tb[:30])
	print wav[a]

	#----------------------------------------- CONTINUOUS EXTINCTION 
	idx =np.argmin(abs(P.h))
	ne=P.ne #h=0 at idx
	Tsun = P.T
	h= P.h
	kappa=P.exthmin(wav*1e4,Tsun[idx],ne[idx]) # wav in AA
	
	'''
	plt.rc('text',usetex=True)
	plt.rc('font',**{'family':'serif','size':14})
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	l1,=ax1.plot(wav,Tb,c='g',label='brightness temperature')
	ax1.set_xlabel(r'wavelength [$\mu$m]')
	ax1.set_ylabel(r'brightness temperature [K]')
	l2,=ax2.plot(wav,kappa/1e-24,'--b',label='H$^-$ extinction')
	ax2.set_ylabel(r'H$^-$ extinction 1E-24[cm$^2$/(H atom)]')
	ax2.invert_yaxis()
	plt.legend(handles=[l1, l2,])
	plt.show()'''

	sigmaT = 6.648e-25 # cm^2 Thomson scattering
	
	nHneutral=P.nH-P.nprot
	HminExt=P.exthmin(0.5*1e4,Tsun,ne)*nHneutral
	labels4=[r'H$^-$ bf+ff extinction','Thomson scattering','Total extinction']
	totalExt = HminExt + sigmaT*ne
	#P.plot_quantities(h,np.array([HminExt,sigmaT*ne,totalExt]),plot_label=labels4,x_label='height [km]', y_label=r'extinction [cm$^{-1]}$]',logY=True)

	#------------------------------- OPTICAL DEPTH

	#idx2 = np.where(wav==0.5)[0]
	#wav500 = wav[np.where(wav==0.5)[0]]*1e4
	tau500 = P.tau500
	tau5 = np.zeros(len(P.tau500))
	for i in range(1,len(tau5)):
		tau5[i]=tau5[i-1]+0.5*(totalExt[i]+totalExt[i-1])*(h[i-1]-h[i])*1e5
	labels5 = [r'observed $\tau_{500}$',r'FALC $\tau_{500}$']
	#P.plot_quantities(h,tau5,plot_label=labels5[0],hold=True)
	#P.plot_quantities(h,tau500,style='--',plot_label=labels5[1],x_label='height [km]', y_label=r'optical depth at $\lambda =500$ nm',logY=True)

	#P.plot_quantities(h,np.abs(tau5-tau500),x_label='height [km]', y_label=r'optical depth difference at $\lambda =500$ nm',logY=True)

	#------------------------ EMERGENT INTENSITY AND HEIGHT FORMATION
	contfunc, intt, hmean, tau, wl = P.emergint(tau5,Tsun,wl=0.5)
	#-------------------- plot signle contribution function
	#P.plot_quantities(h,contfunc/np.max(contfunc),'--',plot_label=r'$\lambda=0.5$')
	
	print 'computed continuum intensity wav =%g : %g erg s-1 cm-2 ster-1 cm-1'%(wl, intt)
	w = np.where(wav == wl)
	print 'observed continuum intensity wav =%g : %g erg s-1 cm-2 ster-1 cm-1' %(wav[w], Icont[w]*1e10*1e4)
	print 'ratio Iobs/comp =',(Icont[w]*1e10*1e4)[0]/intt
	
	#----------------------- plotting of the contribution functions happends in this function for many wavelengths. plot_cont=True
	tau_arr,tauidx,hmean_arr,intt_arr=P.contribution_wav(np.array([.5, 1.,1.6,5.]),wav,h,tau5,Tsun, Icont, plot_cont=False,plot_hmean=False)
	print ' h(tau=1) =', h[tauidx[0]],h[tauidx[1]],h[tauidx[2]], h[tauidx[3]]
	print '<h> =', hmean_arr

	Tb_arr=P.T_brightness(intt_arr,np.array([.5,1.,1.6,5.])/1e4)
	Tsun_arr = np.array([Tsun[tauidx[0]],Tsun[tauidx[1]],Tsun[tauidx[2]],Tsun[tauidx[3]]])

	idx_arr=[np.where(np.abs(Tsun-Tb_arr[i])<80) for i in range(len(tauidx))]

	h_arr = np.array([h[idx_arr[0]],h[idx_arr[1]],h[idx_arr[2]],h[idx_arr[3]]])
	print ' h(Tb=T(h)) =', [[h_arr[i,0],h_arr[i,1]] for i in range(4)]
	print ' T(tau=1) =', Tsun_arr
	print ' Tb =', Tb_arr
	print ' I =', intt_arr
	print ' B =', P.planck(Tsun_arr, np.array([.5,1.,1.6,5.])/1e4)
	print ' I/B =',intt_arr/P.planck(Tsun_arr, np.array([.5,1.,1.6,5.])/1e4)

	#-----------------------------  DISK CENTER INTENSITY
	#P.plot_quantities(wav,Icont,plot_label='observed $I^c_\lambda$',hold=True)
	intt_arr = P.emergint_wav(wav,tau5,Tsun)
	#P.plot_quantities(wav,intt_arr/1e14,plot_label='FALC $I^c_\lambda$',y_label='intensity 1E14 [erg cm$^{-2}$s$^{-1}$cm$^{-1}$ster$^{-1}$]', x_label='wavelength [$\mu$m]')

	#-------------------------------- LIMB DARKENING
	mu = np.linspace(0.1,1,10)
	wavlist = np.linspace(wav[0],wav[-1],10)
	labels6 = [(r'$\lambda=$ %.2f' %iwav) for iwav in wavlist]
	labels7 = [(r'$\mu=$%.2f '%imu) for imu in mu]
	rRsun = np.sin(np.arccos(mu))
	intmu_arr = P.emergint_wav(wavlist,tau5,Tsun,mu)
	intmu_norm = np.array([intmu_arr[i]/intmu_arr[i,-1] for i in range(len(wavlist))])

		
	#P.plot_quantities(mu,intmu_norm,plot_label=labels6, x_label=r'$\mu=\cos\theta$', y_label='intensity')
	#P.plot_quantities(rRsun,intmu_norm, plot_label=labels6, x_label=r'$r/R_\odot=\sin\theta$', y_label='intensity')
	
	for mu_ in np.arange(0.1,1.1,0.1):
		intmu_arr = P.emergint_wav(wav,tau5,Tsun,mu_)
		#intmu_norm = np.array([intmu_arr[i]/intmu_arr[i,-1] for i in range(len(wavlist))])
		'''
		plt.plot(wav,intmu_arr/1e14, label=('$\mu=$'+str(mu_)))
	plt.legend()
	plt.xlabel(r'wavelength $[\mu m]$')
	plt.ylabel('intensity 1E14 [erg cm$^{-2}$s$^{-1}$cm$^{-1}$ster$^{-1}$]')
	plt.tight_layout()
	plt.show()'''
	
	#--------------------------------- FLUX INTEGRATION
	#P.flux_int(wav,Fcont,tau5)
if __name__=='__main__':
	main()
