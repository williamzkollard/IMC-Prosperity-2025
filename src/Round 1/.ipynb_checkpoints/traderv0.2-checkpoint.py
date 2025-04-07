#RAINFOREST_RESIN 
#Price has been stable throughout history

#KELP
#Price has gone up and down throughout history (mean reverting)

#both have position limits of 50 



from datamodel import Listing, OrderDepth, Trade, TradingState

timestamp = 1100

listings = {
	"RAINFOREST_RESIN ": Listing(
		symbol="RAINFOREST_RESIN ", 
		product="RAINFOREST_RESIN ", 
		denomination = "SEASHELLS"
	),
	"KELP": Listing(
		symbol="KELP", 
		product="KELP", 
		denomination = "SEASHELLS"
	),
}

order_depths = {
	"RAINFOREST_RESIN ": OrderDepth(
		buy_orders={10: 7, 9: 5},
		sell_orders={12: -5, 13: -3}
	),
	"KELP": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),	
}

own_trades = {
	"RAINFOREST_RESIN ": [
		Trade(
			symbol="RAINFOREST_RESIN ",
			price=11,
			quantity=4,
			buyer="SUBMISSION",
			seller="",
			timestamp=1000
		),
		Trade(
			symbol="RAINFOREST_RESIN ",
			price=12,
			quantity=3,
			buyer="SUBMISSION",
			seller="",
			timestamp=1000
		)
	],
	"KELP": [
		Trade(
			symbol="KELP",
			price=143,
			quantity=2,
			buyer="",
			seller="SUBMISSION",
			timestamp=1000
		),
	]
}

market_trades = {
	"RAINFOREST_RESIN": [],
	"KELP": []
}

position = {
	"RAINFOREST_RESIN": 10,
	"KELP": -7
}

observations = {}
traderData = ""

state = TradingState(
	traderData,
	timestamp,
  listings,
	order_depths,
	own_trades,
	market_trades,
	position,
	observations
)












class Trader:


    def price_rainforest_resin(self):
        print("hello")

    def price_kelp(self):
        print("hello")


    def _get_orders_rainforest_resin(self, state: TradingState):

        bid_price = 8 # need function to price
        bid_amt = 5
        ask_price = 10 # need function to price
        ask_amt = 5

        return [
            Order("RAINFOREST_RESIN", int(bid_price), bid_amt),
            Order("RAINFOREST_RESIN", int(ask_price), ask_amt),
            ]
        


    def _get_orders_kelp(self, state: TradingState):

        bid_price = 8 # need function to price
        bid_amt = 5
        ask_price = 10 # need function to price
        ask_amt = 5

        return [
            Order("KELP", int(bid_price), bid_amt),
            Order("KELP", int(ask_price), ask_amt),
            ]


    def run(self, state: TradingState):

        orders = {} #initialises the dictionary 
        conversions = 0
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

    
    
    
print(state.order_depths)
        

