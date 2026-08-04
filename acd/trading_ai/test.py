import yfinance as yf

ticker = "AVGO"

data = yf.download(
    ticker,
    start="2022-03-22",
    end="2022-03-23",
    interval="1h",
    prepost=True,
    auto_adjust=False,
)

print(data)