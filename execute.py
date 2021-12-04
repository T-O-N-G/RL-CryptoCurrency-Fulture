import logging
from binance_f import SubscriptionClient
from binance_f.constant.test import *
from binance_f.model import *
from binance_f.exception.binanceapiexception import BinanceApiException

from binance_f.base.printobject import *
# import gc

logger = logging.getLogger("binance-futures")
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

sub_client = SubscriptionClient(api_key=g_api_key, secret_key=g_secret_key)
ticker_last_price = 0.0
orderbooks = []
ticker = []
depth = []
time_type = []

# https://oapi.dingtalk.com/robot/send?access_token=8bd78539539ee4fe42e671be13287813802dddaad79d0bad0b1e268883aec156


def callback(data_type: 'SubscribeMessageType', event: 'any'):
    global ticker_last_price
    global orderbooks
    global ticker
    global depth
    global time_type

    if data_type == SubscribeMessageType.RESPONSE:
        print("Event ID: ", event)
    elif data_type == SubscribeMessageType.PAYLOAD:
        orderbook=[]
        if event.eventType == "24hrTicker":
            ticker=[]
            ticker.extend(["Ticker", event.lastPrice, event.lastQty])

            if ticker_last_price > event.lastPrice:
                ticker.append(event.lastPrice * event.lastQty * -1)
            else:
                ticker.append(event.lastPrice * event.lastQty)

            time_type = [event.eventTime, "ticker_update"]

            ticker_last_price = event.lastPrice

        if event.eventType == "depthUpdate":
            depth=[]
            depth.append("Depth")

            for i in range(20):
                depth.append(event.bids[i].price)
            for i in range(20):
                depth.append(event.bids[i].qty)
            for i in range(20):
                depth.append(event.asks[i].price)
            for i in range(20):
                depth.append(event.asks[i].qty)

            time_type = [event.eventTime, "depth_update"]

        orderbook.extend(time_type)
        orderbook.extend(depth)
        orderbook.extend(ticker)

        if len(orderbooks)>10:
            orderbooks.pop(0)
        orderbooks.append(orderbook)
        del orderbook
        # print(orderbooks)
        for i in range(len(orderbooks)):
            print(orderbooks[i][0])

    else:
        print("Unknown Data:")


def error(e: 'BinanceApiException'):
    print(e.error_code + e.error_message)


sub_client.subscribe_symbol_ticker_event("bnbusdt", callback, error)
sub_client.subscribe_book_depth_event(
    "bnbusdt", 20, callback, error, update_time=UpdateTime.NORMAL)
# sub_client.subscribe_book_depth_event("bnbusdt",20,callback,error,update_time="@250ms")
