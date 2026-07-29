import time
import threading
import warnings
from datetime import datetime
from typing import Dict , Optional
import pandas as pd # type: ignore
#### ib dependencies
from ibapi.client import EClient # type: ignore
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from util.appenv import APPENV # this is for the custom environment variables and configuration


class TradingApp(EClient,EWrapper,APPENV):
    def __init__(self)->None:
        EClient.__init__(self,self)
        APPENV.__init__(self)
        self.data: Dict[int,pd.DataFrame]={}
    def error(self , reqId ,errorTime,errorCode, errorString, advanceOrderReject):
        print(f'reqId: {reqId}, time: {errorTime}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advanceOrderReject}')
        
    def get_historical_data(self,reqId:int,contract:Contract)->pd.DataFrame:
        self.data[reqId]=pd.DataFrame(columns=['time','high','low','close'])
        self.data[reqId].set_index('time',inplace=True)
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime='',
            durationStr='365 D',
            barSizeSetting='1 hour',
            whatToShow='MIDPOINT',
            useRTH=0,
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )
        time.sleep(4)
        return self.data[reqId]
    def parse_ib_date(value):

        value = str(value)

        # Unix timestamp from IB
        if value.isdigit() and len(value) == 10:
            return pd.to_datetime(int(value), unit='s')

        # IB formatted date
        return pd.to_datetime(value)
    def historicalData(self, reqId:int, bar:BarData)->None:
        df=self.data[reqId]
        # timestamp = self.parse_ib_date(bar.date)
        
        df.loc[bar.date,['open','high','low','close','volume']]=[bar.open,bar.high,bar.low,bar.close,bar.volume]
        df=df.astype(float)
        self.data[reqId]=df
 
    @staticmethod
    def get_stock_contract(symbol:str)->Contract:
        contract=Contract()
        contract.symbol=symbol
        contract.secType='STK'
        contract.exchange='SMART'
        contract.currency='USD'
        return contract
    
    
app=TradingApp()
app.connect(app.host, app.port, app.client_id)