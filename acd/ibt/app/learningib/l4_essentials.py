import random

from ibapi.client import *
from ibapi.wrapper import *
import time 
import threading
from util.appenv import APPENV
class TestApp (EClient,EWrapper,APPENV):
    def __init__(self):
        EClient.__init__(self,self)
        APPENV.__init__(self)

    def nextValidId(self,orderId):
        self.orderId=orderId
    
    def nextId(self):
        self.orderId +=1
        return self.orderId
    def disconnect_properly(self):
        """Make sure we disconnect properly to release the clientId."""
        self.disconnect()
        print(f"Disconnected and clientId {self.clientId} is released.")
    def wait_for_orderId(self):
        """Block the thread until nextValidId has been called to initialize orderId."""
        while self.orderId is None:
            time.sleep(.4)  # Sleep for a short period and check again
    def currentTime(self, time):
        print(time)     
    
    def error(self , reqId ,errorTime,errorCode, errorString, advanceOrderReject):
        print(f'reqId: {reqId}, time: {errorTime}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advanceOrderReject}')
        
app=TestApp()
app.connect(app.host, app.port, app.client_id)

threading.Thread(target=app.run).start()
time.sleep(1)
# Wait until we have a valid orderId
app.wait_for_orderId()

for i in range(0,5):
    print(app.nextId())
    app.reqCurrentTime()