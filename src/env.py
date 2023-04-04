import requests
import json


class TradingEnv:
    def __init__(self, window_size=10, steps=100):
        self.current_step = window_size
        self.window_size = window_size
        self.steps = steps
        self.bought_price = 0
        self.sold_price = 0
        self.bought = False

        self.reset()

    def reset(self):
        self.data = json.loads(requests.get('http://192.168.50.100:6000/binance/btcusdt').text).values()
        self.dataDict = json.loads(requests.get('http://192.168.50.100:6000/binance/btcusdt').text)
        self.current_step = self.window_size
        self.bought_price = 0
        self.sold_price = 0
        self.bought = False
        return self.data