#RAINFOREST_RESIN 
#Price has been stable throughout history

#KELP
#Price has gone up and down throughout history (mean reverting)

#both have position limits of 50 



from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from typing import List
import string


#Trade logic:
#Simply view the market price (from order depths and do +1 -1 of the price)

class Trader:

    def price_rainforest_resin(self, state: TradingState):
        if 'RAINFOREST_RESIN' in state.order_depths:
            buy_orders = list(state.order_depths['RAINFOREST_RESIN'].buy_orders.items())
            sell_orders = list(state.order_depths['RAINFOREST_RESIN'].sell_orders.items())

            if buy_orders and sell_orders:
                market_price_rainforest_resin = (buy_orders[0][0] + sell_orders[0][0]) / 2
                return market_price_rainforest_resin
            else:
                raise ValueError("Buy orders or sell orders are empty for RAINFOREST_RESIN")
        else:
            raise KeyError("RAINFOREST_RESIN not found in order_depths")


    def price_kelp(self, state: TradingState):
        if 'KELP' in state.order_depths:
            buy_orders = list(state.order_depths['KELP'].buy_orders.items())
            sell_orders = list(state.order_depths['KELP'].sell_orders.items())

            if buy_orders and sell_orders:
                market_price_kelp = (buy_orders[0][0] + sell_orders[0][0]) / 2
                return market_price_kelp
            else:
                raise ValueError("Buy orders or sell orders are empty for KELP")
        else:
            raise KeyError("KELP not found in order_depths")


    def _get_orders_rainforest_resin(self, state: TradingState):

        bid_price = self.price_rainforest_resin(state) -1 
        bid_amt = 5
        ask_price = self.price_rainforest_resin(state) + 1
        ask_amt = -5

        return [
            Order("RAINFOREST_RESIN", int(bid_price), bid_amt),
            Order("RAINFOREST_RESIN", int(ask_price), ask_amt),
            ]


    def _get_orders_kelp(self, state: TradingState):

        bid_price = self.price_kelp(state) -1
        bid_amt = 5
        ask_price = self.price_kelp(state) +1
        ask_amt = -5

        return [
            Order("KELP", int(bid_price), bid_amt),
            Order("KELP", int(ask_price), ask_amt),
            ]


    def run(self, state: TradingState):

        orders = {} #initialises the dictionary 
        conversions = 1
        trader_data = ""
    
        #self._update_data(state)

        # get orders for RAINFOREST_RESIN
        order_rainforest_resin = self._get_orders_rainforest_resin(state)
        if order_rainforest_resin:
            orders["RAINFOREST_RESIN"] = order_rainforest_resin

        # get orders for KELP
        order_kelp = self._get_orders_kelp(state)
        if order_kelp:
            orders["KELP"] = order_kelp


        print(f"Orders FINAL: {orders}")



        return orders, conversions, trader_data


    


        

