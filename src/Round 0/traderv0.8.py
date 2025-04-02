#RAINFOREST_RESIN 
#Price has been stable throughout history

#KELP
#Price has gone up and down throughout history (mean reverting)

#both have position limits of 50 



from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from typing import List
import string
import numpy as np
import pandas as pd 




class Trader:

    def __init__(self):
        #store order book data from the tradingstate
        #specifically it will store the current mid price and spread of eacch product
        #this data will come from the update_data function
        #each iteration, more data is added (ie from the TradingState order depths)
        self.rainforest_resin = [] 
        self.kelp = []


    def _vw_mid_price(self, data):
        data = {i: abs(data[i]) if not np.isnan(data[i]) else 0 for i in data}

        bid_amt = data["bid_volume_1"] + data["bid_volume_2"] + data["bid_volume_3"]
        ask_amt = data["ask_volume_1"] + data["ask_volume_2"] + data["ask_volume_3"]
        tot_amt = bid_amt + ask_amt

        vw_bid = (
            data["bid_price_1"] * data["bid_volume_1"]
            + data["bid_price_2"] * data["bid_volume_2"]
            + data["bid_price_3"] * data["bid_volume_3"]
        ) / bid_amt
        vw_ask = (
            data["ask_price_1"] * data["ask_volume_1"]
            + data["ask_price_2"] * data["ask_volume_2"]
            + data["ask_price_3"] * data["ask_volume_3"]
        ) / ask_amt

        s = vw_ask * bid_amt / tot_amt + vw_bid * ask_amt / tot_amt

        return s
    

    def _update_data(self, state: TradingState):
        """
        Update the prices data in the object
        instance, recording the last prices
        """

        order_depth = state.order_depths
        for product, orders in order_depth.items():
            data: dict[str, Time | float] = {
                "timestamp": state.timestamp,
                'ask_volume_1': np.nan,
                'ask_volume_2': np.nan,
                'ask_volume_3': np.nan,
                'bid_volume_1': np.nan,
                'bid_volume_2': np.nan,
                'bid_volume_3': np.nan,
                'ask_price_1': np.nan,
                'ask_price_2': np.nan,
                'ask_price_3': np.nan,
                'bid_price_1': np.nan,
                'bid_price_2': np.nan,
                'bid_price_3': np.nan,
            }

            for i, limit in enumerate(orders.buy_orders, 1):
                amount = orders.buy_orders[limit]

                data[f"bid_price_{i}"] = limit
                data[f"bid_volume_{i}"] = amount


            for i, limit in enumerate(orders.sell_orders, 1):
                amount = orders.sell_orders[limit]

                data[f"ask_price_{i}"] = limit
                data[f"ask_volume_{i}"] = amount

            data['mid_price'] = (data["ask_price_1"] + data["bid_price_1"]) / 2
            data["spread"] = (data["ask_price_1"] - data["bid_price_1"]) / data["mid_price"]
            data['mid_price_vw'] = self._vw_mid_price(data)


            if product == "RAINFOREST_RESIN":
                self.rainforest_resin.append(data) #data is a dictionary 
            elif product == "KELP":
                self.kelp.append(data)


    def _stoikov_bidask(
            self, 
            mid_price_vw: int,  # mid price
            current_pos: int,  # current position
            target_pos: int,  # target position
            timestamp: int,  # timestamp
            gamma: float ,  # risk aversion
            sigma: float,  # volatility of price
            k: float
            ):
        
        q = current_pos - target_pos

        total_timestamps = 2000000
        T = 1
        t = timestamp / total_timestamps
        sigma = np.sqrt(sigma)  # convert variance to std dev

        # calculate ref price using Stoikov model
        ref_price = mid_price_vw - q * gamma * sigma ** 2 * (T-t)

        stoikov_spread = gamma * sigma ** 2 * (T - t) + 2 / gamma * np.log(1 + gamma / k)

        bid, ask = ref_price - stoikov_spread / 2, ref_price + stoikov_spread / 2
        return np.floor(bid), np.ceil(ask)    
                


    def _get_orders_rainforest_resin(self, state: TradingState):
        
        
        prices = pd.DataFrame.from_records(self.rainforest_resin)
        mid_price_vw = prices['mid_price_vw'].iloc[-1]
        current_pos = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
        t = state.timestamp

        # shared parameters for Stoikov-Anelladeva Model
        gamma = 0.025
        k = 0.5
        sigma = pd.to_numeric(
            prices["mid_price_vw"].rolling(50, min_periods=5).var().iloc[-1]
        )
        if np.isnan(sigma):
            sigma = 2.2402965654304166

        bid, ask = self._stoikov_bidask(mid_price_vw, current_pos, 0, t, gamma, sigma, k)


        bid_amt = min(50 - current_pos, 10)
        ask_amt = -min(50 + current_pos, 10)


        orders = [
            Order("RAINFOREST_RESIN", int(ask), ask_amt),
            Order("RAINFOREST_RESIN", int(bid), bid_amt),
        ]

        return orders
 

    def _get_orders_kelp(self, state: TradingState):
        
        prices = pd.DataFrame.from_records(self.kelp)
        mid_price_vw = prices['mid_price_vw'].iloc[-1]
        current_pos = state.position["KELP"] if "KELP" in state.position else 0
        t = state.timestamp

        # shared parameters for Stoikov-Anelladeva Model
        gamma = 0.025
        k = 0.5
        sigma = pd.to_numeric(
            prices["mid_price_vw"].rolling(50, min_periods=5).var().iloc[-1]
        )
        if np.isnan(sigma):
            sigma = 2.2402965654304166

        bid, ask = self._stoikov_bidask(mid_price_vw, current_pos, 0, t, gamma, sigma, k)

        bid_amt = min(50 - current_pos, 10)
        ask_amt = -min(50 + current_pos, 10)


        orders = [
            Order("KELP", int(ask), ask_amt),
            Order("KELP", int(bid), bid_amt),
        ]
        return orders



    def run(self, state: TradingState):

        orders = {} #initialises the dictionary 
        conversions = 1
        trader_data = ""
    
        self._update_data(state)

        # get orders for RAINFOREST_RESIN
        order_rainforest_resin = self._get_orders_rainforest_resin(state)
        if order_rainforest_resin:
            orders["RAINFOREST_RESIN"] = order_rainforest_resin

        # get orders for KELP
        order_kelp = self._get_orders_kelp(state)
        if order_kelp:
            orders["KELP"] = order_kelp


        return orders, conversions, trader_data



    










