""" small general purpose helpers """
# was grtls.py

import sys,os,math
#import logging
import re
#from typing import Tuple

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

def unixsecond(julianday):
	''' convert julianday (UTC) to seconds since actual unix epoch '''
	if sys.implementation.name == "micropython":
		return int((julianday - 2440587.5 - 10957.5) * 86400.0)  #  (2000-1970)*365.25
	else: 
		return (julianday - 2440587.5) * 86400.0

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

def prettyprint(fetchrecs) -> None:
	''' print the records fetched by fetch method to the console '''
	for tpl in fetchrecs:
		tm = prettydate(tpl[0])   
		print("%s %4.3g %s %s" % (tm,tpl[1],tpl[2],tpl[3]))

def graphyprint(fetchrecs, xcol=0, ycol=1) -> None:
	''' print graphically to console selected quantity trace from database '''
	curve = [rec[ycol] for rec in fetchrecs]
	printCurve(curve)
	xnms = [rec[xcol] for rec in fetchrecs]
	#print("xnms:{}".format(min(xnms)))
	if xnms:
		minx = min(xnms)
		maxx = max(xnms)
		if minx>10000: # assumed juliandays
			printTimeAx(xnms)
		else:
			printNumAx(xnms,round((maxx-minx)/4,1)) 

def printTimeAx(jddata) -> None:
	''' print juliandate time x axis to console with | as day separator '''
	def diffxh(julday, hr24=0):
		""" frac day diff noon """
		julday -= 0.5 # now jd starts at midnight
		julday += hr24/24.0
		julday -= timezone / 60 / 60 / 24  # go to local time
		return abs(round(julday)-julday)  # midnight is 0
	
	minx=jddata[0]
	maxx=jddata[-1]
	if maxx-minx>9:
		xstp = 7 # a week
		itm  = 1 # mnth
	elif maxx-minx>1:
		xstp=1  # a day
		itm =2  # mday
	else:
		xstp =1/24.0 # an hour
		itm =3  # hour
	nminx = (minx//xstp)*xstp  # normalise to multiples of xstp
	stps = [nminx+i*xstp+xstp for i in range(int((maxx-minx)//xstp))]
	
	print('graph min:{} {} max:{} {} xstp={} nstps={} today@{}'.format( nminx,miny,maxx,maxy, xstp,len(stps),utcnow))

	noon=-3
	print(chr(0x2595),end='')
	print("=",end='')
	for i in range(len(jddata)-2):
		df = [diffxh(jd) for jd in jddata[i:i+3]]
		if df.index(min(df))==1: # mid close to midnight
			print("|",end='')
			#logger.debug("marker@%s df:%s jd=%.5f" % (prettydate(jddata[i+1]),df,jddata[i+1]))
		elif df.index(max(df))==1: # mid close to noon
			print(prettydate(jddata[i+1],"{:%02d}"),end='')
			noon=i+1
		elif i>noon: # skip lbl size
			print("-",end='')
	print("=")

	
def printNumAx(nmdata, interv=0.5):  # TODO!!
	pos=0
	print(chr(0x2595),end='')
	print(chr(0x2594),end='')
	refx = int(nmdata[0] / interv) * interv + interv
	sepx = refx-interv/2
	ofsx = interv*0.0
	if sepx<nmdata[0]:
		sepx+=interv
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
	return
	"""		
		if df.index(max(df))==1:
			print("|",end='')
		elif df.index(min(df))==1:
			print("{}".format(nmdata[i+1]),end='')
			noon=i+1
		elif i>noon+1:
			print("-",end='')
	print("=")
	"""

def printCurve(data, height=10, vmax=None, vmin=None, backgndchar=0x2508) -> None:
	''' print float data array graphically to console using block char fillings '''
	if data is None or len(data)==0:
		#logger.error("no data to graph")	
		return
	if vmax is None: 
		vmax = max(data)
	if vmin is None: 
		vmin = min(data)
	if vmax==vmin:
		sf=1.0
	else:
		sf = (height-1)/(vmax-vmin)
	#logger.info("curve min=%f max=%f sf=%f" % (vmin,vmax,sf))
	for ln in range(height-1,-1,-1):  # 9..0
		print(chr(0x2595),end='')
		for y in data:
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
	pass
	#logger = logging.getLogger(__name__)	# get logger from main program
print("\nbye")