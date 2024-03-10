import math
"""/// <summary>
    /// numerical recipes in pascal chapter 9.3 Van Wingaarden-Dekker-Brent method
    /// used for finding dominant wavelength
    """
class BrentRootFinder():
	maxIter = 100;
	eps = 3e-8  # float precision
	def __init__(self,function, xMin=0, xMax=1):
		self.func = function
		self.iter = 0;
		self.tol = BrentRootFinder.eps
		self.a = xMin #//fa = function(a);
		self.b = xMax #//fb = function(b);
		self.c = xMax
		#self.e = math.nan
		self.fa = function(xMin) #math.nan
		self.fb = function(xMax) #math.nan
		self.fc = self.fb
        
	def __str__(self):
		return "RootFnd{}: f({:.2f})={:.4g} f({:.2f})={:.4g}".format(self.iter, self.a,self.fa, self.b,self.fb)
	def Proposition(self, x):
		fx = self.func(x)
		if (self.fa * fx < 0):
			self.a = x
			self.fa = fx
			return True
        
		if (self.fb * fx < 0):
			self.b = x
			self.fb = fx
			return True
		return False


	def SetStart(self, _a, _b):
		self.a = _a
		self.fa = self.func(_a)
		self.b = _b 
		self.c = _b
		self.fb = self.func(_b)
		self.fc = self.fb
		if (self.fa * self.fb > 0.0):
			logger.warning("root must be bracketed")
			return False    #error: root must be bracketed
		else:
			return True

	def Solve0(self, xStart):
		if (math.isnan(self.fa)):
			self.fa = self.func(self.a)
		if (math.isnan(self.fb)):
			self.fb = self.func(self.b);
		xi = xStart;
		nxi = math.nan
		nyi = math.nan
		nextStep = True
		breakpoint()
		while (nextStep and self.iter < BrentRootFinder.maxIter):
			yi = nyi
			nyi = self.func(xi)
			xi = nxi
			nxi,nextStep = self.Predict(nyi)
			#self.fb = function(self.b)
		return xi,yi

	def Solve(self):
		nextStep=True
		self.iter=0
		nb=self.b
		while (nextStep and self.iter < BrentRootFinder.maxIter):
			nfb = self.func(nb)
			nb,nextStep = self._Predict(nb, nfb)
		return self.b, self.fb

	def _Predict(self, nb, nfb):
		""" propose a new x setpoint to get fx closer to 0 """
		nextStep=False
		#double min1, min2, min, p, q, r, s, xm, tol1;
		if nfb is None: # or len(nfb)==0 or math.isnan(nfb[0]):
			logger.warning("supply funcVal fb:{}".format(nfb))
		if (nfb * self.fc > 0.0):  # no 0 between b-c 
			self.c = self.a		# take a for c
			self.fc = self.fa
			d = nb - self.a	# adjust bounding interval d
			self.e = d
		else: 
			#c = self.b
			#fc = self.fb
			d = self.b - self.a # 240303
		self.b = nb
		self.fb = nfb
        
		if (math.fabs(self.fc) < math.fabs(self.fb)):  # if c was better : discard a 
			self.a = self.b	# take b for a
			self.b = self.c	# take c for b
			self.c = self.a
			self.fa = self.fb
			self.fb = self.fc
			self.fc = self.fa
		# now b is best guess, c=a is second
        
		tol1 = 2.0 * BrentRootFinder.eps * math.fabs(self.b) + 0.5 * self.tol  # small step
		xm = 0.5 * (self.c - self.b)
		if (math.fabs(xm) <= tol1 or self.fb == 0.0): # convergence check
			nextStep = False
			logger.debug("root {}:{} xm={}".format('found' if math.fabs(self.fb)<self.tol else 'NOT found',self.b,xm))
			#return b
		else:
			nextStep = True
			self.iter += 1
			if (math.fabs(self.e) >= tol1 and math.fabs(self.fa) > math.fabs(self.fb)): 
				#attempt inverse quadratic interpolation
				s = self.fb / self.fa
				if (self.a == self.c):
					p = 2.0 * xm * s
					q = 1.0 - s
				else:
					q = self.fa / self.fc
					r = self.fb / self.fc
					p = s * (2.0 * xm * q * (q - r) - (self.b - self.a) * (r - 1.0))
					q = (q - 1.0) * (r - 1.0) * (s - 1.0)
				if (p > 0.0):
					q = -q #  // check whether in bounds
				p = math.fabs(p)
				min1 = 3.0 * xm * q - math.fabs(tol1 * q)
				min2 = math.fabs(self.e * q)
				if (min1 < min2): 
					mn = min1 
				else:
					mn = min2
				if (2.0 * p < mn): #accept interpolation
					self.e = d
					d = p / q
				else: # failed, use bisection
					d = xm
					self.e = d
			else: # bounds decreasing too slowly, use bisection
				d = xm
				self.e = d
				logger.debug("bisect:b+{}".format(d))
			logger.debug("ntry:{} a:{} b:{}".format(self.iter,self.a,self.b))
			self.a = self.b  # // move last best guess to a
			self.fa = self.fb
			nb = self.b
			if (math.fabs(d) > tol1):
				nb += d  # // eval new trial root
			else:
				if (xm > 0):
					nb += math.fabs(tol1)
				else:
					nb -= math.fabs(tol1)
		return nb, nextStep

if __name__ == "__main__":
	def func(wl):
		return math.sin(wl)
	import logging
	from tls import get_logger
	#set_logger(level=logging.INFO)
	#logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
	logger = get_logger(__file__, logging.DEBUG)
	brt = BrentRootFinder(func)
	brt.SetStart(-1,3.1) # having some bisect
	x,y = brt.Solve()
	logger.info("around 0:{} Solved:{} -> {}".format(0,x,y))
	#brt.SetStart(0.1, 3.1) # not bracked
	brt.SetStart(-0.1, 3.13)
	x,y = brt.Solve()
	logger.info("around 0:{} Solved:{} -> {}".format(0,x,y))
	#brt.SetStart(3.15, 6.3)
	brt.SetStart(4*math.pi-0.3, 4*math.pi+1.5) # only bisect
	x,y = brt.Solve()
	logger.info("around 4pi:{} Solved:{} -> {}".format(4*math.pi,x,y))
	brt.SetStart(-3.15,-2) # beyond xmin xmax initially set
	x,y = brt.Solve()
	logger.info("-pi, Solved:{} -> {}".format(x,y))
	
else:
	import logging
	logger = logging.getLogger(__name__)
	


