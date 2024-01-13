""" small general purpose helpers """
# was grtls.py

import sys,os,math
#import logging
import re
#from typing import Tuple
from collections import namedtuple

if sys.implementation.name == "micropython":
	from utime import time,gmtime	# type: ignore
	import machine				# type: ignore
	_S_DELTA = 946681200		# 2000-1-1  to 1970-1-1	=10957 days
	timezone = -3600  # CET ofs for local time
else:
	import datetime
	from time import time,timezone	# type ignore
	_S_DELTA = 0

	def seconds_since_epoch(epoch = datetime.datetime.utcfromtimestamp(0), utcnow=datetime.datetime.utcnow()):
		''' time in s since 1970-1-1 midnight utc'''
		return (utcnow - epoch).total_seconds()
	
_MJD = float(2400000.5)

def find_min_index(lst):
	min_index = min(range(len(lst)), key=lambda i: lst[i])
	return min_index

def localtime(julianday):
	""" time struct incl DST and TZ """
	return time.localtime(unixsecond(julianday))

def julianday(tunix = None, isMJD=False) -> float:
	''' convert unix time i.e.  to julianday i.e. days since noon on Monday, January 1 4713 BC
		tunix can be either a float as returned by time.time() or a tuple as returned by 	'''
	if tunix is None:
		if sys.implementation.name == "micropython":
			if sys.platform=="linux":
				tunix = gmtime()  # (Y,M,D,H,M,S,,)  utc
			else:
				tunix = list(machine.RTC().datetime()) # tuple (Y,M,D,wd,H,M,S,subs)
				del tunix[3]
		else:
			tunix = time()   # float seconds since 00:00:00 Thursday, 1 January 1970
		#return julianday(tunix, isMJD)
	if isinstance(tunix ,(tuple,list)):
		#_io=1 if sys.implementation.name == "micropython" else 0
		#print("tupjd={}".format(tunix))
		Y,M,D = tunix[0:3]
		JDN = -_MJD if isMJD else 0.0
		mm = -1 if M<=2 else 0  #  math.trunc((M-14)/12)  # 0 or -1
		JDN += (1461 * (Y + 4800 + mm))//4 
		#if isMJD:
		#	JDN -= _MJD
		JDN += (367 *(M - 2 - 12 * mm)) //12 
		JDN -= (3 * ((Y + 4900 + mm)//100))//4 
		JDN += D - 32075
		#H,MM,S=tunix[4:7] if sys.implementation.name == "micropython" else tunix[3:6]
		H,MM,S=tunix[3:6]
		#if isMJD:
		#	JDN += (H)/24
		#else:
		JDN += (H-12)/24
		JDN += MM/1440
		JDN += S/86400
		print('Y-M-D H:M:S {}-{}-{} {}:{}:{}'.format(Y,M,D,H,MM,S))
		return JDN 
	elif _S_DELTA:
		tunix += _S_DELTA
	if isMJD:
		return (tunix / 86400.0) + 40587.0  # epoch midnight on November 17, 1858.
	return (tunix / 86400.0 ) + 2440587.5  # epoch noon on Monday, January 1 4713 BC

def _JulianDay (Year,Month,Day, Hour=0,Minute=0,Second=0) -> float:
    """calculate number of days since beg of year -4712  (=JDN)
       Astron.Algorith chap.7
    """
    if Month==1 or Month==2:
        Month += 12
        Year -= 1
    JD = (Hour + (Minute + Second/60.0) /60) /24
    JD += math.floor(365.25 * (Year + 4716)) + \
          math.floor(30.6001 * (Month+1)) + \
          Day - 1524.5
    if Year>1582 or \
       (Year==1582 and
        (Month>10 or (Month==10 and Day>4))):
       A = math.floor(Year/100.0)
       JD += 2.0 - A + math.floor(A/4)
    return JD

def JulianDay (gmtime):
    return _JulianDay(gmtime[0],gmtime[1],gmtime[2],\
                      gmtime[3],gmtime[4],gmtime[5])



def unixsecond(julianday):
	''' convert julianday (UTC) to seconds since actual unix epoch '''
	if sys.implementation.name == "micropython":
		return int((julianday - 2440587.5 - 10957.5) * 86400.0)  #  (2000-1970)*365.25
	else: 
		return (julianday - 2440587.5) * 86400.0

def JulianTime(jdn, isMJD=False): # -> Tuple[int,int,int,int,int,float]:
	""" get (Y M D H M S) from julianday number see https://quasar.as.utexas.edu/BillInfo/JulianDatesG.html """
	if isMJD:
		F, I = math.modf(jdn)
		A = math.trunc((I + 532784.25)/36524.25)  # 2400000.5 - 1867216.25 = 532784.25
		I += 2400001
		#if F>0.5:
		#	I+=1
	else:
		jdn+=0.5 # => midnight
		F, I = math.modf(jdn)
		A = math.trunc((I - 1867216.25)/36524.25)
	if I > 2299160:
		B = I + 1 + A - math.trunc(A / 4.)
	else:
		B = I
	C = B + 1524
	D = math.trunc((C - 122.1) / 365.25)
	E = math.trunc(365.25 * D)
	G = math.trunc((C - E) / 30.6001)
	day = C - E + F - math.trunc(30.6001 * G)
	if G < 13.5:
		month = G - 1
	else:
		month = G - 13
	if month > 2.5:
		year = D - 4716
	else:
		year = D - 4715	
	hour = F % 1 *24
	minute = hour % 1 * 60
	second = minute % 1 * 60	
	return year,month,int(day),int(hour),int(minute),second


def TimeTup(JulianDay):  # (Y,M,D,H,M,S,wd,0,0)
    """ return python time struct
        Meeus Astron.Algorith chap.7
        note : not effective with mktime below year 1970
    """
    F,Z = math.modf(JulianDay+0.5)
    wd = int(Z % 7)
    if Z<2299161:
        A=Z
    else:
        alp = math.floor((Z-1867216.25)/36524.25)
        A = Z+1+alp-math.floor(alp/4)
    B=A+1524
    yy=int((B-122.1)/365.25)
    D = math.floor(365.25*yy)
    mo = int((B-D)/30.6001)
    dd = B-D-math.floor(30.6001*mo)+F
    if mo<14:
        mo -=1
    else:
        mo -=13
    if mo>2:
        yy-=4716
    else:
        yy-=4715
    F=round(F*86400) # whole seconds
    F,ss = divmod(F,60)
    hh,mm = divmod(F,60)
    tup = (yy,mo,int(dd),int(hh),int(mm),int(ss),wd,0,0)
    return tup


def weekday(jd, isMJD=False) -> int:
	""" 0=Sunday """
	if isMJD:
		return (jd+3) % 7
	return (jd+1.5) % 7
	
def prettydate(JulianDay, format="{:%d %H:%M:%S}") -> str:
	''' generates string representation of julianday '''
	if not JulianDay:
		JulianDay=julianday()
		if sys.implementation.name == "micropython":
			if sys.platform=="linux":
				tobj = gmtime() # tuple (Y,M,D,H,M,S,wd,yd)
			else: # "esp32"
				tobj = list(machine.RTC().datetime()) # tuple (Y,M,D,wd,H,M,S,subs)
				del tobj[7]
				del tobj[3]
		else:
			tobj = datetime.datetime.utcnow() #  utc time
	else:
		if sys.implementation.name == "micropython":
			tobj = gmtime(unixsecond(JulianDay))
		else:
			#tobj = TimeTup(JulianDay)
			tobj = datetime.datetime.fromtimestamp(unixsecond(JulianDay))  # named tuple
		#print("tobj={}".format(tobj.hour))
	if format=="#j4":
		fd = int(4*(JulianDay % 1))
		return ('after noon','evening','night','morning')[fd]	
	#print("tobj:{}".format(tobj))
	if sys.implementation.name == "micropython":
		return ("{:02d}".format(tobj[2]))
	#	return("{} {:02d}:{:02d}:{:02d}").format(tobj[2],tobj[4],tobj[5],tobj[6])
	return format.format(tobj)
	return ("{} {}:{}:{}").format(tobj.tm_mday,tobj.tm_hour,tobj.tm_min,tobj.tm_sec)
	return time.strftime(format, tobj)

def SiNumForm(num) -> str:
	''' format number with SI prefixes '''
	pref = ['f','p','n','u','m',' ','k','M','G','T','P','E','Z','Y']
	mul=1e-15
	for pr in pref:
		if abs(num)/mul<999:
			break
		mul *= 1000
	if pr=='k' and num<1900:
		return "{:4.0f} ".format(num)
	return "{:4.3g}{}".format(num/mul,pr)

def normGRdat(fetchrecs, xcol=0, ycol=1):
	''' normalise fetchrecs to have xdat equidistant; missing recs will have nan in ydat '''
	xdat = [rec[xcol] for rec in fetchrecs]
	xmin = min(xdat)
	xmax = max(xdat)
	dx = min([xdat[i+1]-xdat[i] for i in range(len(xdat)-1) if xdat[i+1]>xdat[i]])
	xlen = int((xmax-xmin)/dx)+1
	logger.debug("normalizing GRdat len:{}->{} dx={}".format(len(xdat),xlen,dx))
	dat = { int((x-xmin)/dx):idx for (idx,x) in enumerate(xdat)  }
	ydat = [math.nan] * xlen
	for i,idx in dat.items():
		xorg = xdat[idx]
		x = xmin+i*dx
		if round(x,6)!=round(xorg,6):
			logger.warning("xorg:{}!={}".format(xorg,x))
		if i<len(ydat) and dat[i]<len(fetchrecs):
			ydat[i] = fetchrecs[dat[i]][ycol]
		else:
			logger.warning("i:{}>={} or idx:{}>=len{}".format(i,len(ydat),dat[i],len(fetchrecs)))
	return [(xmin+i*dx, y) for i,y in enumerate(ydat)]
	
def prettyprint(fetchrecs) -> None:
	''' print the records fetched by fetch method to the console '''
	for tpl in fetchrecs:
		tm = prettydate(tpl[0])
		logger.debug("{} {:4.3g} {} {}".format(tm,tpl[1],tpl[2],tpl[3]))
		print("%s %4.3g %s %s" % (tm,tpl[1],tpl[2],tpl[3]))

def graphyprint(fetchrecs, xcol=0, ycol=1) -> None:
	''' print graphically to console selected quantity trace from database '''
	nrmdat = normGRdat(fetchrecs, xcol, ycol)
	curve = [rec[1] for rec in nrmdat]
	printCurve(curve)
	xnms = [rec[0] for rec in nrmdat]
	#print("xnms:{}".format(min(xnms)))
	if xnms:
		minx = min(xnms)
		maxx = max(xnms)
		if minx>10000: # assumed juliandays
			printTimeAx(xnms)
		else:
			printNumAx(xnms,round((maxx-minx)/4,1)) 

tmAx = namedtuple('tmAx',('lbl','avgmin','tmstep','lblformat'), defaults=(u'1day',15,0.25,'#j4'))
""" {ndays:(lbl,avgmin,gridstep,lblformat),} """
tmBACKs={0.2:tmAx(u'5hr',5,0.0417,'{:%H}'),
	1.0:tmAx(u'1day',15,0.25,'#j4'),  #'%H:%M'), 
	5.0:tmAx(u'5days',20,1,'{:%d}'),
	30.44:tmAx(u'1mnth',6*60,7,'wk{:%V}'), 
	182.6:tmAx(u'6mnth',24*60,30.44,'{:%b}'), 
	365.25:tmAx(u'1yr',2*24*60,30.44,'{:%b}') }

def tmTUP(ndays):
	if ndays in tmBACKs:
		return tmBACKs[ndays]
	else:
		for d,tup in tmBACKs.items():
			if d>=ndays:
				return tup
	return tmBACKs[1.0]
	
def printTimeAx(jddata) -> None:
	''' print juliandate time x axis to console with | as day separator '''	
	minx=jddata[0]
	maxx=jddata[-1]
	#ndays = maxx-minx
	if maxx-minx>9:
		xstp = 7.0 # a week
		itm  = 1 # mnth
	elif maxx-minx>1:
		xstp=1.0  # a day
		itm =2  # mday
	else:
		xstp =1.0/24.0 # an hour
		itm =3  # hour
	ndays = maxx-minx
	#if ndays not in tmBACKs:
	#	ndays=1.0
	nminx = (minx//xstp)*xstp  # normalise to multiples of xstp
	stps = [nminx+i*xstp+xstp for i in range(int((maxx-minx)//xstp))]
	xlbltup = tmTUP(ndays) # tmBACKs[ndays]
	lblformat=xlbltup[3]
	gridstep=xlbltup[2]
	avgmin=xlbltup[1]  # barwdt avgminutes = xlbltup[1]
	
	logger.debug('timeAx min:{} max:{} ndays={} xstp:{} nstps={} tup:{}'.format( prettydate(minx),prettydate(maxx), ndays,xstp,len(stps),xlbltup))
	pos=0
	refx = int(minx / xstp) * xstp + xstp
	sepx = refx-gridstep/2
	ofsx = xstp*0.0
	if sepx<jddata[0]:
		sepx+=gridstep
	print(chr(0x2595),end='')
	print(chr(0x2594),end='')
	for i in range(len(jddata)-1):
		if i>pos:
			if jddata[i]>refx+ofsx:
				#nms = "{:.3}".format(nmdata[i])
				nms = prettydate(refx,lblformat) # "{:%02d}") #"{:03.3g}".format(refx)
				print(nms,end='')
				refx += gridstep
				pos+=len(nms)-1
			elif jddata[i]>sepx:
				print(chr(0x2579),end='') # tic mark
				sepx += gridstep
			elif i>pos:
				print(chr(0x2594),end='')
			pos+=1
	print(chr(0x2594))
	
def printNumAx(nmdata, interv=0.5):  # TODO!!
	pos=0
	print(chr(0x2595),end='')
	print(chr(0x2594),end='')
	refx = int(nmdata[0] / interv) * interv + interv
	sepx = refx-interv/2
	ofsx = interv*0.0
	if sepx<nmdata[0]:
		sepx+=interv
	logger.debug("numXax nx:{} refx:{} interv:{}".format(len(nmdata),refx,interv))
	for i in range(len(nmdata)-1):
		#df = [(nm % interv) for nm in nmdata[i:i+3]]
		if i>pos:
			if nmdata[i]>refx+ofsx:
				#nms = "{:.3}".format(nmdata[i])
				nms = "{:03.3g}".format(refx)
				print(nms,end='')
				refx += interv
				pos+=len(nms)-1
			elif nmdata[i]>sepx:
				print(chr(0x2579),end='') # tic mark
				sepx += interv
			elif i>pos:
				print(chr(0x2594),end='')
			pos+=1
	print(chr(0x2594))
	
def printCurve(ydat, height=10, vmax=None, vmin=None, backgndchar=0x2508) -> None:
	''' print float data array graphically to console using block char fillings '''
	if ydat is None or len(ydat)==0:
		#logger.error("no data to graph")	
		return
	if vmax is None: 
		vmax = max(ydat)
	if vmin is None: 
		vmin = min(ydat)
	if vmax==vmin:
		sf=1.0
	else:
		sf = (height-1)/(vmax-vmin)
	logger.debug("curve min={} max={} scale={}".format(vmin,vmax,sf))
	for ln in range(height-1,-1,-1):  # 9..0
		print(chr(0x2595),end='') # left Yax
		for y in ydat:
			if math.isnan(y):
				print(chr(backgndchar),end='')
				continue
			lny = (y-vmin)*sf
			if ln < lny-1:
				print(chr(0x2588),end='')  # bar body
			elif ln < lny:
				print(chr(0x2581+int((lny-ln)*8.0)),end='')	 # top of bar
			else:
				print(chr(backgndchar),end='')  # grid
		print(chr(0x258f),"{:03.3g}".format(vmin+ 1/sf/2 + ln/sf))  # labels y-ax

if __name__ == "__main__":
	if sys.implementation.name != "micropython":
		import pdb
	from tls import get_logger
	logger = get_logger(__name__)
	print("{}-{} tz:{}s".format(sys.implementation.name,  os.uname().sysname if 'uname' in dir(os) else sys.platform, timezone))
	NAIS = (1961,3,12, 15,10,0,0)
	jdhj = julianday(NAIS)
	print(' JD: hj={} = jt:{} on wd:{}'.format(jdhj,JulianTime(jdhj),weekday(jdhj)))
	jdhj = julianday(NAIS,isMJD=True)
	print('MJD: hj={} = jt:{} on wd:{}'.format(jdhj,JulianTime(jdhj,isMJD=True),weekday(jdhj,True)))
	
	nw = time() 
	jdnow = julianday(nw)
	jdt = julianday()
	print('unixnow={} jdnow={} jdt={} wd={} prdate={}'.format(nw,jdnow,jdt,weekday(jdt),prettydate(None)))
	
	jdnow = julianday(nw,True)
	jdt = julianday(isMJD=True)
	print('MJDnow= jdnow={} jdt={} wd={} jt={}'.format(jdnow,jdt,weekday(jdt,True),JulianTime(jdt,True)))
	if sys.implementation.name != "micropython":
		pdb.set_trace()
	
	recs = []
	ndays = 2
	utcofs = timezone /60/60/24 # days ofs
	jd0 = julianday()  #_JulianDay(2023,12,1)
	for h in range(-24*ndays,1):
		jd = (h/24.0) #+ jd0   # ndays back till jd0 local time
		F,Z = math.modf(jd+0.5-utcofs) # F=0 is midnight local time
		recs.append((jd, math.sin(math.pi*(F-0.25)*2.0), F*24))
	print("recs:{}".format(recs))
	graphyprint(recs)
else:
	import logging
	logger = logging.getLogger(__name__)	# get logger from main program
print("\nbye")