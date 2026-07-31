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
import re
import time
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
from config import DATA_DIR

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
        # ADD THIS LINE TO FIX THE ATTRIBUTE ERROR:
        self.earliest_date = {} 


    # --------------------------------------------------
    # IB Next Valid order ID callback
    # --------------------------------------------------

    def nextValidId(self, orderId):

        self.next_order_id = orderId

        print(f"IB connected. Next valid order ID: {orderId}")

        self.connected_event.set()

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
        if not self.connected_event.wait(timeout=10):
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
        if stock_symbol.lower()=='bit' or stock_symbol.lower()=='bitcoin':
           contract.exchange='PAXOS'
        elif overnight:
            contract.exchange = "OVERNIGHT"
        else:
            contract.exchange = "SMART"
            contract.primaryExchange = "NASDAQ"
        contract.currency = "USD"
        if stock_symbol=='COMP':
            contract.secType='IND'
        if stock_symbol.lower()=='vix':
            contract.exchange='SMART'
            contract.primaryExchange='CBOE'
     
        return contract
# --------------------------------------------------
# Historical Data callbacks
# --------------------------------------------------

    def historicalData(self, reqId, bar: BarData):
        print("BAR:", bar.date)
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
            self.earliest_date[reqId] = None

        self.historical_data[reqId].append({
            "datetime": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": float(bar.volume)
        })
        # Track the oldest timestamp in this specific chunk sequence
        self.earliest_date[reqId] = bar.date

    def historicalDataEnd(self, reqId, start, end):

        print(f"Historical download complete ({reqId})")

        if reqId in self.request_events:
            self.request_events[reqId].set()
# --------------------------------------------------
    # Helper to parse human-readable durations into total days
    # --------------------------------------------------
    def _parse_duration_to_days(self, duration_str):
        match = re.match(r"(\d+)\s*([WwDdMmYy])", duration_str.strip())
        if not match:
            return 365 # Default fallback
        
        val = int(match.group(1))
        unit = match.group(2).upper()
        
        if unit == 'D': return val
        if unit == 'W': return val * 7
        if unit == 'M': return val * 30
        if unit == 'Y': return val * 365
        return 365

    # --------------------------------------------------
    # Core multi-year historical collector
    # --------------------------------------------------
    def get_historical_data(self, symbol, duration=duration, bar_size=bar_size, what_to_show="TRADES"):
        total_days_needed = self._parse_duration_to_days(duration)
        days_remaining = total_days_needed
        
        all_dfs = []
        end_date_anchor = "" 
        
        print(f"Total days requested: {total_days_needed}. Executing chunk sequence...")

        while days_remaining > 0:
            # Step down cleanly by maximum allowable chunk lengths to reduce unnecessary loops
            chunk_days = min(days_remaining, 365)
            chunk_duration_str = f"{chunk_days} D"
            
            reqId = self.req_counter
            self.req_counter += 1

            self.historical_data[reqId] = []
            self.earliest_date[reqId] = None

            event = threading.Event()
            self.request_events[reqId] = event

            contract = self.stock_contract(symbol)
            print(f"Sending request ID {reqId} for {chunk_duration_str} back from anchor: '{end_date_anchor}'...")
            
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime=end_date_anchor,
                durationStr=chunk_duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                # Controlled directly via commands: OVERNIGHT forces useRTH=False. SMART daytime uses True.
                useRTH=int(False) if overnight else int(True),
                formatDate=1, 
                keepUpToDate=False,
                chartOptions=[]
            )
            
            if not event.wait(timeout=45):
                print(f"Historical data request {reqId} timed out. Processing collected data.")
                self._cleanup_request(reqId)
                break

            chunk_df = pd.DataFrame(self.historical_data[reqId])
            last_seen_date = self.earliest_date[reqId]
            self._cleanup_request(reqId)

            if chunk_df.empty:
                print(f"No more data returned by IBKR for chunk ID {reqId}. Terminating loop walkback.")
                break

            all_dfs.append(chunk_df)
            days_remaining -= chunk_days

            if days_remaining > 0:
                if not last_seen_date:
                    break
                
                # --- CRITICAL FIX FOR DUPLICATION ---
                # Daily bars are returned without timestamps (e.g. '20250212'). 
                # This logic converts either date format into a standard IB API compatible timestamp string.
                date_str = str(last_seen_date).strip()
                if " " in date_str:
                    clean_date = date_str.split()[0]
                    clean_time = date_str.split()[1]
                    end_date_anchor = f"{clean_date}-{clean_time}"
                else:
                    # Clean up standard formats like 'YYYYMMDD' or 'YYYY-MM-DD' cleanly into a midnight anchor
                    clean_date = date_str.replace("-", "")
                    end_date_anchor = f"{clean_date}-00:00:00"

                time.sleep(1.5)

        if not all_dfs:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        final_df = pd.concat(all_dfs, ignore_index=True)
        
        if not final_df.empty:
            final_df["datetime"] = pd.to_datetime(final_df["datetime"], errors='coerce')
            final_df = final_df.dropna(subset=["datetime"])
            
            # --- CRITICAL BULLETPROOF DEDUPLICATION INSURANCE ---
            final_df = final_df.drop_duplicates(subset=["datetime"], keep="first")
            final_df = final_df.sort_values(by="datetime").reset_index(drop=True)

        return final_df
    def _cleanup_request(self, reqId):
        if reqId in self.historical_data: del self.historical_data[reqId]
        if reqId in self.request_events: del self.request_events[reqId]
        if reqId in self.earliest_date: del self.earliest_date[reqId]

try:
    app=IBClient()
    app.connect_ib()

    df = app.get_historical_data(symbol=stock_symbol, duration=duration,
                                 bar_size=bar_size)
    print("\n--- OUTPUT SNAPSHOT ---")
    print(df.head(2))
    print(df.tail(2))
    print(f"Total entries loaded: {len(df)}")
    
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    if(len(df) > 0):
        df.to_csv(f"{DATA_DIR}/{str(stock_symbol).lower()}_{clean_bar_name}_{ 'overnight' if overnight else 'init' }.csv", index=False)
    else:
        print("No data received. CSV file not created.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    try:
        app.disconnect()
    except:
        pass
