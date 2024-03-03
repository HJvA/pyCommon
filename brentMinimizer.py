import math,sys
    #/// <summary>
    #/// from numerical recipes in pascal  page 321
    #/// </summary>
class BrentMinimizer():
	""" https://en.m.wikipedia.org/wiki/Brent%27s_method """
	#private func function;
	#private int itmax;

	#double a, b, d, e;
	#double fv, fw, fx;
	#iter = 0
	#double tol1, tol2;
	#double v, w, x, xm;

	""" /// <summary>
        /// 
        /// </summary>
        /// <param name="Function">user function </param>
        /// <param name="xMin">X interval lowest</param>
        /// <param name="xMax">X interval upest</param>
        /// <param name="maxIter">max nr of iterations</param>
	"""
	def __init__(self, func, xMin, xMax):
		self.function = func
		#self.itmax = maxIter;
		#self.iter = 0;
		if (xMin < xMax): 
			self.a = xMin
		else:
			self.a = xMax  # a,b  in Ascending order
		if (xMin > xMax): 
			self.b = xMin 
		else: 
			self.b = xMax
		self.fv = math.inf 
		self.v = math.nan
		self.fw = math.inf
		self.w = math.nan
		self.fx = math.inf 
		self.x = math.nan
		self.d = 0.0 
		self.e = 0.0
	
	def __str__(self):
		return "v:{}->{} w:{}-> x:{}->{}".format(self.v,self.fv, self.w,self.fw, self.x,self.fx)

	""" /// <summary>
        /// supply first possible values as start for iteration
        /// </summary>
        /// <param name="bx"></param>  """
	def SetStart(self, bx): 
		#v = bx; w = v; x = v; d = 0; e = 0; fx = function(x); fv = fx; fw = fx;
		self.EvalProposition(bx, self.function(bx))

	def sign(self, a, b):
		if (b >= 0): 
			return math.fabs(a)
		else: 
			return -math.fabs(a);
        
	def GetPrediction(self, xm, tol1, tol2):
		cgold = 0.3819660
		u = 0.0 # local
		if (math.fabs(self.e) > tol1):  #do not try a step too small 
			# construct a trial parabolical fit throug x,v,w }
			r = (self.x - self.w) * (self.fx - self.fv)
			q = (self.x - self.v) * (self.fx - self.fw)
			p = (self.x - self.v) * q - (self.x - self.w) * r
			q = 2 * (q - r)
			if (q > 0):
				p = -p
			q = math.fabs(q)
			etemp = self.e;
			self.e = self.d;
			if (math.fabs(p) >= math.fabs(0.5 * q * etemp) or p <= q * (self.a - self.x) or p >= q * (self.b - self.x)): #// reject fit?
				#   {take golden section step in largest segment}
				if (self.x >= xm):
					self.e = self.a - self.x
				else:
					self.e = self.b - self.x
				self.d = cgold * self.e  #;  // golden section
			else: # // accept fit when step in a,b interval and impies a movement less then half the movement of step before last 
				#{   //{parabolic fit accepted}
				self.d = p / q
				u = self.x + self.d
				if (u - self.a < tol2 or self.b - u < tol2):
					self.d = self.sign(tol1, xm - self.x)
		else: # take golden section step in largest segment
			if (self.x >= xm):
				self.e = self.a - self.x
			else:
				self.e = self.b - self.x
			self.d = cgold * self.e
			#}  //END; {if}
		#// d either from golden section or from parablolic fit
		if (math.fabs(self.d) > tol1):
			u = self.x + self.d
		else:
			u = self.x + self.sign(tol1, self.d)
		return u
        

	def EvalProposition(self,  u,  fu):
		if (fu <= self.fx): #  // better then x
			if (not math.isnan(self.x)):
				if (u >= self.x):
					self.a = self.x
				else:
					self.b = self.x            
			self.v = self.w 
			self.fv = self.fw
			self.w = self.x
			self.fw = self.fx
			self.x = u
			self.fx = fu
		else:
			if (u < self.x):
				self.a = u
			else:
				self.b = u
			if (fu <= self.fw or self.w == self.x):
				self.v = self.w 
				self.fv = self.fw
				self.w = u
				self.fw = fu
			elif (fu <= self.fv or self.v == self.x or self.v == self.w):
				self.v = u
				self.fv = fu
			else:
				return False
		#self.iter+=1
		return True;
   

	""" /// <summary>
        /// purpose : find minimum of user function func
        ///| from    : numerical recipes in pascal page 321
        ///| input   : ax _lt bx _lt cx AND func(ax) _gt func(bx) _lt func(cx)
        ///|         : tol : requested precision
        ///| output  : xmin,ymin : coordinates of min of user function
        /// </summary>
        /// <param name="bx">best initial X guess (inside interval)</param>
        /// <param name="tol"></param>
        /// <param name="xmin"></param>
        /// <param name="ymin"></param>
        /// <returns>nr of iterations</returns> """
	def brent(self, bx, tol, itrmax=100):  #ref xmin, ref ymin):
            #//const int itmax = 100;    //    {max iterations}
		itr =0
		zeps = 1e-20 #//      {protect for brent=0}
            #// the x for least so far fx is bracketed between a and b
            #// w has second least fw so far, v has previous value of w
            #// u is most recent
		self.SetStart(bx)
		if (itr < 3):
			self.SetStart(self.a)
		if (itr < 3):
			self.SetStart(self.b)
		xmin = math.nan 
		ymin = math.nan
		while (itr < itrmax):
			xm = 0.5 * (self.a + self.b)
			tol1 = tol * math.fabs(self.x) + zeps
			tol2 = 2 * tol1
			if (math.fabs(self.x-xm) <= tol2 - 0.5 * (self.b-self.a)):  #//   { test stop }
				break
			else:
				u = self.GetPrediction(xm, tol1, tol2)
				fu = self.function(u)
				self.EvalProposition(u, fu)
				itr+=1
				logger.debug("brent: u={} du={}".format(u,fu))
		xmin = self.x
		ymin = self.fx
		return xmin,ymin;

if __name__ == "__main__":	
	#from stdRadiators import Daylight
	#sys.path.append('.')
	#import submod.pyCommon.tls as tls
	import tls
	logger = tls.get_logger(__file__)
	import numpy as np
	def func(x):
		return math.sin(x)
	brt = BrentMinimizer(func, 0, 6.28)
	brt.SetStart(math.pi)
	brt.SetStart(math.pi+0.1)
	x,y = brt.brent(0, 0.0001)
	logger.info("brent :{} -> {}  ({})".format(x,y, math.pi*1.5))
else:
	import logging
	logger = logging.getLogger(__name__)

