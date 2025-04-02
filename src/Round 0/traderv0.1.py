from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Trade
from typing import List
import string



timestamp = 1100

listings = {
	"PRODUCT1": Listing(
		symbol="PRODUCT1", 
		product="PRODUCT1", 
		denomination= "SEASHELLS"
	),
	"PRODUCT2": Listing(
		symbol="PRODUCT2", 
		product="PRODUCT2", 
		denomination = "SEASHELLS"
	)
}

order_depths = {
	"PRODUCT1": OrderDepth(
		buy_orders={10: 7, 9: 5},
		sell_orders={12: -5, 13: -3}
	),
	"PRODUCT2": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),	
}

own_trades = {
	"PRODUCT1": [
		Trade(
			symbol="PRODUCT1",
			price=11,
			quantity=4,
			buyer="SUBMISSION",
			seller="",
			timestamp=1000
		),
		Trade(
			symbol="PRODUCT1",
			price=12,
			quantity=3,
			buyer="SUBMISSION",
			seller="",
			timestamp=1000
		)
	],
	"PRODUCT2": [
		Trade(
			symbol="PRODUCT2",
			price=143,
			quantity=2,
			buyer="",
			seller="SUBMISSION",
			timestamp=1000
		),
	]
}

market_trades = {
	"PRODUCT1": [],
	"PRODUCT2": []
}

position = {
	"PRODUCT1": 10,
	"PRODUCT2": -7
}

observations = {}
traderData = ""


class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData


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


trader = Trader()
iteration = trader.run(state)

iteration