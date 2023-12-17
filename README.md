# pyCommon
common library for python projects on multiple platforms 

## targets  

### micropyhon  
'''  
if sys.implementation.name == "micropython":
	from utime import time	# type: ignore
	import machine				# type: ignore
	if sys.platform=="esp32":
		print(machine.RTC().datetime())
'''
''' bash pi4
https://forums.raspberrypi.com/viewtopic.php?t=191744
cd ~
sudo apt install libffi-dev
git clone https://github.com/micropython/micropython.git  
cd micropython
git pull
cd ports/unix
make clean
make axtls
cd ../../mpy-cross
make
cd ../ports/unix
make submodules 
make
sudo ln -s ~/micropython/ports/unix/build-standard/micropython /usr/local/bin/micropython
cd ~
micropython -h
'''
### windows  
'''  
if sys.platform=="win32": # os.name=="nt"
   import adodbapi
   dsn=dbFname #'dsCATdat'
   provider='MSDASQL'  # ODBC for OLEDB
   dbms = adodbapi.connect('Data Source=%s;Provider=%s;'% (dsn,provider))
            wildcard='%%'
 

'''
### symbian  
''' 
if sys.platform=="symbian_s60":
    import dbS60
    if '--dat' in sys.argv:
        datPath = sys.argv[sys.argv.index('--dat')+1]
    ConStr = os.path.join(datPath, dbFname) #r'stars.db')
    dbms = dbS60.connect(ConStr)
'''
### linux  
'''
if sys.platform=="linux":
    from sqlite3 import connect,OperationalError
    store = os.path.join(datPath, dbFname) # r'stars.db')
    store = os.path.expanduser(store) 
    if not os.path.isfile(store): 
        dbms=connect(store, check_same_thread=False)
        if dbms:
            dbms=dbCreate(dbms)


'''
