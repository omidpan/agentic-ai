# filename: ib_client.py

"""
Interactive Brokers Historical Data Client

Features:
- IB API background network thread
- nextValidId handshake
- historicalDataEnd completion signaling
- no sleep-based waiting
- OHLCV normalization
- reusable stock contract builder

Requires:
    pip install ibapi pandas
    example for run: python ibq_sraper.py -s NVDA -d "1 D" -o True -bs "1 min"
"""
import argparse
import random
import threading
import pandas as pd
from datetime import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

from utils.appenv import APPENV
from config import DATA_DIR,INTERVAL,PERIOD

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument("-s" ,"--symbol",
                    type=str,
                    required=False,
                    help="The stock symbol to process. default is NVDA.", 
                    nargs='?', default="NVDA")
parser.add_argument("-o", "--overnight",
                    type=bool,
                    required=False,
                    help="Overnight or daily. default is SMART.", 
                    nargs='?', default=False)
parser.add_argument("-d", "--duration",
                    type=str,
                    required=False,
                    help="Duration of historical data. default is 365 D.", 
                    nargs='?', default="365 D")
parser.add_argument("-bs", "--bar_size",
                    type=str,
                    required=False,
                    help="candle size of historical data. default is 1 hour.", 
                    nargs='?', default="1 hour")
args = parser.parse_args()

# 4. Access the value using dot notation
stock_symbol = args.symbol.upper()
overnight = args.overnight
duration = args.duration
bar_size = args.bar_size
class IBClient(EWrapper, EClient, APPENV):

    def __init__(self):

        EClient.__init__(self, self)
        APPENV.__init__(self)

        self.next_order_id = None

        self.connected_event = threading.Event()

        self.req_counter = random.randint(1, 10000)

        self.historical_data = {}
        self.request_events = {}


    # --------------------------------------------------
    # IB Next Valid order ID callback
    # --------------------------------------------------

    def nextValidId(self, orderId):

        self.next_order_id = orderId

        print(f"IB connected. Next valid order ID: {orderId}")

        self.connected_event.set()



    def error(self,reqId,errorCode,errorString,advancedOrderRejectJson=""):

        print(f"""
IB ERROR
Request ID : {reqId}
Code       : {errorCode}
Message    : {errorString}
"""
        )
        if reqId in self.request_events and errorCode in [162, 200, 321]:
            self.request_events[reqId].set()



    # --------------------------------------------------
    # Connection management
    # --------------------------------------------------

    def connect_ib(self):

        print(f"Connecting IB {self.host}:{self.port}")
        self.connect(self.host,self.port,self.client_id)
        api_thread = threading.Thread(target=self.run,daemon=True)

        api_thread.start()
        if not self.connected_event.wait(timeout=15):
            raise ConnectionError("IB API handshake failed")
        print("IB API ready")


    # --------------------------------------------------
    # Contract builder
    # --------------------------------------------------

    @staticmethod
    def stock_contract(symbol):

        contract = Contract()

        contract.symbol = symbol

        contract.secType = "STK"
        if overnight:
            contract.exchange = "OVERNIGHT"
        else:
            contract.exchange = "SMART"
        

        contract.currency = "USD"
     
        return contract
# --------------------------------------------------
# Historical Data callbacks
# --------------------------------------------------

    def historicalData(self, reqId, bar: BarData):
        print("BAR:", bar.date)
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []

        self.historical_data[reqId].append({
            "datetime": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": float(bar.volume)
        })


    def historicalDataEnd(self, reqId, start, end):

        print(f"Historical download complete ({reqId})")

        if reqId in self.request_events:
            self.request_events[reqId].set()
            
                
    def get_historical_data(
                self,
                symbol,
                duration=duration,
                bar_size=bar_size,
                what_to_show="TRADES",
                useRTH=True
        ):

            reqId = self.req_counter
            self.req_counter += 1

            self.historical_data[reqId] = []

            event = threading.Event()
            self.request_events[reqId] = event

            contract = self.stock_contract(symbol)
            print(contract)
            print("Sending historical request...")
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime = "",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=int(useRTH),
                formatDate=2, # 1 is yyyymmdd{space}{space}hh:mm:dd, 2 is unix timestamp
                keepUpToDate=False,
                chartOptions=[]
            )
            print("Historical request sent.")
            if not event.wait(timeout=60):
                raise TimeoutError("Historical data request timed out.")

            df = pd.DataFrame(self.historical_data[reqId])

            del self.historical_data[reqId]
            del self.request_events[reqId]

            # if not df.empty:
            #     df["datetime"] = pd.to_datetime(df["datetime"])

            return df


app=IBClient()
app.connect_ib()

df = app.get_historical_data(symbol=stock_symbol, duration=duration,
                             bar_size=bar_size, 
                             useRTH=False)

print(df.head())

print(df.tail())

print(len(df))
if(len(df) > 0):
    df.to_csv(f"{DATA_DIR}/{str(stock_symbol).lower()}_{bar_size}_init.csv", index=False)
else:
    print("No data received. CSV file not created.")
    

app.disconnect()
