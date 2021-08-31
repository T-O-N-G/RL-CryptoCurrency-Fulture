import logging
from binance_f import SubscriptionClient
from binance_f.constant.test import *
from binance_f.model import *
from binance_f.exception.binanceapiexception import BinanceApiException

from binance_f.base.printobject import *

logger = logging.getLogger("binance-futures")
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

sub_client = SubscriptionClient(api_key=g_api_key, secret_key=g_secret_key)
line_ticker = ""
ticker_last_price = 0.0

line_depth = ""
eventTime = ""


def callback(data_type: 'SubscribeMessageType', event: 'any'):
    global line_ticker
    global line_depth
    global eventTime
    global ticker_last_price

    if data_type == SubscribeMessageType.RESPONSE:
        print("Event ID: ", event)
    elif data_type == SubscribeMessageType.PAYLOAD:
        if event.eventType == "24hrTicker":
            line_ticker = "Ticker, " + str(event.lastPrice) + ", " + str(event.lastQty) + ", "
            if ticker_last_price > event.lastPrice:
                line_ticker += str(event.lastPrice * event.lastQty * -1)
            else:
                line_ticker += str(event.lastPrice * event.lastQty)
            eventTime = str(event.eventTime) + ", ticker_update, "
            ticker_last_price = event.lastPrice
        if event.eventType == "depthUpdate":
            line_depth = "Depth, "
            for i in range(20):
                line_depth += str(event.bids[i].price)+", "
            for i in range(20):
                line_depth += str(event.bids[i].qty)+", "
            for i in range(20):
                line_depth += str(event.asks[i].price)+", "
            for i in range(20):
                line_depth += str(event.asks[i].qty)+", "
            eventTime = str(event.eventTime) + ", depth_update, "
        print(eventTime + line_depth + line_ticker)
    else:
        print("Unknown Data:")
    # print()


def error(e: 'BinanceApiException'):
    print(e.error_code + e.error_message)


sub_client.subscribe_symbol_ticker_event("ethusdt", callback, error)
sub_client.subscribe_book_depth_event("ethusdt", 20, callback, error, update_time=UpdateTime.NORMAL)
# sub_client.subscribe_book_depth_event("ethusdt", 20, callback, error, update_time="@500ms")
