### base on the pyindi example
# custom for lx200 could work with other mounts to
import sys
import time
import logging
# import the PyIndi module
import PyIndi

# The IndiClient class which inherits from the module PyIndi.BaseClient class
# It should implement all the new* pure virtual functions.
class IndiClient(PyIndi.BaseClient):
    def __init__(self):
        super(IndiClient, self).__init__()
        self.logger = logging.getLogger('IndiClient')
        self.logger.info('creating an instance of IndiClient')
    def newDevice(self, d):
        self.logger.info("new device " + d.getDeviceName())
    def newProperty(self, p):
        self.logger.info("new property "+ p.getName() + " for device "+ p.getDeviceName())
    def removeProperty(self, p):
        self.logger.info("remove property "+ p.getName() + " for device "+ p.getDeviceName())
    def newBLOB(self, bp):
        self.logger.info("new BLOB "+ bp.name.decode())
    def newSwitch(self, svp):
        self.logger.info ("new Switch "+ svp.name + " for device "+ svp.device)
    def newNumber(self, nvp):
        self.logger.info("new Number "+ nvp.name + " for device "+ nvp.device)
    def newText(self, tvp):
        self.logger.info("new Text "+ tvp.name + " for device "+ tvp.device)
    def newLight(self, lvp):
        self.logger.info("new Light "+ lvp.name + " for device "+ lvp.device)
    def newMessage(self, d, m):
        self.logger.info("new Message "+ d.messageQueue(m))
    def serverConnected(self):
        self.logger.info("Server connected ("+self.getHost()+":"+str(self.getPort())+")")
    def serverDisconnected(self, code):
        self.logger.info("Server disconnected (exit code = "+str(code)+","+str(self.getHost())+":"+str(self.getPort())+")")

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.WARNING)

class LX200MountGuide():
    def __init__(self, host = "localhost", port = 7624):
        self.indiclient = IndiClient()
        self.telescope = None
        self.host = host
        self.port = port

    def connect(self):
        self.indiclient.setServer(self.host, self.port)
        print("Connecting and waiting 1 sec")
        if (not(self.indiclient.connectServer())):
            print("No indiserver running on " + self.indiclient.getHost() + ":" +str(self.indiclient.getPort()))
            return False

        time.sleep(1)
        self.telescope = self.connect_and_getdevice("LX200 Classic")
        if self.telescope is None:
            print("Telescope connection error")
            self.indiclient.disconnectServer()
            return False
        return True

    def disconnect(self):
        self.indiclient.disconnectServer()

    def connect_and_getdevice(self, device:str):
        device_handle = self.indiclient.getDevice(device)
        timeout_start = time.time()
        while not(device_handle):
            time.sleep(0.5)
            device_handle = self.indiclient.getDevice(device)
            if time.time() > timeout_start + 5: # 5 seconds
                return None

        device_connect=device_handle.getSwitch("CONNECTION")
        while not(device_connect):
            time.sleep(0.5)
            device_connect=device_handle.getSwitch("CONNECTION")
            if time.time() > timeout_start + 5: # 5 seconds overall
                return None

        if not(device_handle.isConnected()):
            device_connect[0].s=PyIndi.ISS_ON  # the "CONNECT" switch
            device_connect[1].s=PyIndi.ISS_OFF # the "DISCONNECT" switch
            self.indiclient.sendNewSwitch(device_connect)

        return device_handle

    def guide(self, guidedirection="W", guidetime=100):
        if self.telescope is None:
            #try to connect if connect issn't called 
            if self.connect() == False:
                return None
        
        timeout_start = time.time()
        timeout = 2

        ns_numbers = self.telescope.getNumber("TELESCOPE_TIMED_GUIDE_NS")
        while not ns_numbers:
            time.sleep(0.5)
            ns_numbers = self.telescope.getNumber("TELESCOPE_TIMED_GUIDE_NS")
            if time.time() > timeout_start + timeout: # timeout seconds overall
                return None

        we_numbers = self.telescope.getNumber("TELESCOPE_TIMED_GUIDE_WE")
        while not we_numbers:
            time.sleep(0.5)
            we_numbers = self.telescope.getNumber("TELESCOPE_TIMED_GUIDE_WE")
            if time.time() > timeout_start + timeout: # timeout seconds overall
                return None

        for directions in [we_numbers, ns_numbers]:
            for direction in directions:
                if direction.name == "TIMED_GUIDE_" + guidedirection:
                    direction.value = guidetime
                elif "TIMED_GUIDE_" in direction.name:
                    direction.value = 0
        
        self.indiclient.sendNewNumber(we_numbers)
        self.indiclient.sendNewNumber(ns_numbers)

        return True