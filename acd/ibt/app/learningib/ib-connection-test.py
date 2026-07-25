from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from threading import Thread
import logging
from util.appenv import APPENV
log=logging.getLogger(__name__)
class IBClient(EWrapper,EClient):
    def __init__(self,host,port,client_id):
        EClient.__init__(self,self)
        self.connect(host,port,clientId=client_id)
        thread=Thread(target=self.run)
        thread.start()
    def error(self , reqId ,errorTime,errorCode, errorString, advanceOrderReject):
        if errorCode in [2104,2106,2158]:
                    print(errorCode)
                    print(errorString)
        else:
                    # log(f'msg {code}')
                    print('Error {}: {}'.format(errorCode,errorString))
        print(f'reqId: {reqId}, time: {errorTime}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advanceOrderReject}')

# client
env=APPENV()
client=IBClient(env.host, env.port, env.client_id)