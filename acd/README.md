# How to Enable and Start TWS API and IB Gateway

## Step 1: Configure IB Gateway Settings

After installing **IB Gateway**, verify that the date, time, and connection settings are correctly configured.

### Required Actions:
1. Open **IB Gateway**.
2. Check the system date and time settings.
3. Review the API connection settings.
4. Make sure the API ports are configured manually and are **not using the default values**.
5. Save the configuration changes.

> **Note:** Using custom ports helps avoid conflicts with other applications and provides better control when integrating with trading systems.

---

## Step 2: Download and Install TWS API

Download the **Trader Workstation API (TWS API)** package for your specific operating system and machine architecture.

### Download Link:
[Interactive Brokers TWS API Download](https://interactivebrokers.github.io/#)

### Installation Steps:
1. Download the appropriate TWS API package.
2. Extract the downloaded ZIP file.
3. Navigate to the extracted directory IBJts/source/pythonclinet
4. make sure you have virtual environment before installation
5. run the following command ` python setup.py install` to install python dependencies and libs
6. After successful depencies resolve, execute the following command to see the installation result ` pip show ibapi `

---

## Step 3: Folder structure for TWSAPI
### There are source 



### Video Tutorial:
Watch the following tutorial for a complete setup walkthrough:

[How to Setup TWS API - YouTube](https://www.youtube.com/watch?v=MfInGtsh8LY)

---

## Verification Checklist

Before connecting your trading application, confirm the following:

- [ ] IB Gateway is installed successfully.
- [ ] IB Gateway API settings are enabled.
- [ ] Custom API ports are configured.
- [ ] TWS API package has been downloaded and extracted.
- [ ] API libraries are available for your development environment.
- [ ] A test connection between your application and IB Gateway is successful.

---
# how to check kafka topics and group running inside docker
## check first  groups
`docker exec -it local_kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list`
## docker tracking for messages
`docker exec -it local_kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list
`
## to analyze the health of message consumer and producer
`docker exec -it local_kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group trading_strategy_group`
## These are most useful libraries in python for indicators and chart for data . 
### conda install -c conda-forge mplfinance or pip install mplfinance for renko and kagi
## conda install -c conda-forge pandas-ta or  pip install pandas_ta for (indicator like MACD,RSI ATR,..) https://github.com/aarigs/pandas-ta
# pip install plotly  for better visualization, like Kagi and Renko
## Next Steps

After completing this setup, you can proceed with developing your trading application using the IB API connection for:

- Real-time market data streaming
- Order submission and management
- Account and portfolio monitoring
- Automated trading workflows

