#RAINFOREST_RESIN 
#Price has been stable throughout history

#KELP
#Price has gone up and down throughout history (mean reverting)

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

# Define the available products for trading
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

# Define parameters for each product, including fair value estimation and trading strategy settings
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,  # Estimated fair price for trading decisions
        "take_width": 1,  # Price width within which we aggressively take orders
        "clear_width": 0,  # Threshold for clearing orders
        "disregard_edge": 1,  # Ignore orders too close to fair value
        "join_edge": 2,  # Join existing orders within this range
        "default_edge": 4,  # Default spread if no other condition applies
        "soft_position_limit": 10,  # Soft limit on how many units we hold before adjusting strategy
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,  # Avoid large trades that could move the market against us
        "adverse_volume": 15,  # Maximum volume to avoid for adverse selection
        "reversion_beta": -0.229,  # Mean reversion parameter
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        # Define position limits for each product
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50}

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        """
        Executes aggressive trades by taking the best available market orders within a given threshold.
        :param product: The product being traded
        :param fair_value: Estimated fair value of the asset
        :param take_width: Price width within which to take orders
        :param orders: List of orders to be placed
        :param order_depth: Current market order book
        :param position: Current position held
        :param buy_order_volume: Volume of buy orders already placed
        :param sell_order_volume: Volume of sell orders already placed
        :param prevent_adverse: Flag to prevent taking large adverse trades
        :param adverse_volume: Max volume allowed to avoid adverse selection
        :return: Updated buy and sell order volumes
        """
        position_limit = self.LIMIT[product]

        # Check if there are sell orders available to buy from
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())  # Lowest selling price
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]  # Available volume to buy

            # Avoid taking orders that are too large (adverse selection prevention)
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)  # Max buyable amount
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Check if there are buy orders available to sell to
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())  # Highest buying price
            best_bid_amount = order_depth.buy_orders[best_bid]  # Available volume to sell

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)  # Max sellable amount
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        Places limit orders to make the market by providing liquidity at bid and ask prices.
        :param product: The product being traded
        :param orders: List of orders to be placed
        :param bid: Price at which we are willing to buy
        :param ask: Price at which we are willing to sell
        :param position: Current position held
        :param buy_order_volume: Volume of buy orders already placed
        :param sell_order_volume: Volume of sell orders already placed
        :return: Updated buy and sell order volumes
        """
        # Determine how much we can still buy within the position limit
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Place buy order

        # Determine how much we can still sell within the position limit
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Place sell order

        return buy_order_volume, sell_order_volume
    
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        """
        Clears existing positions by placing orders at a fair price with a given width.
        
        - If the trader has a net positive position, it places sell orders at a fair price + width.
        - If the trader has a net negative position, it places buy orders at a fair price - width.
        """
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Determine total volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Determine total volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """
        Computes the fair value of KELP using market mid-prices and a mean-reversion model.
        
        - If the order book is non-empty, it calculates the mid-price.
        - Adjusts the fair price based on a reversion model if historical data is available.
        """
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Filter orders that exceed the adverse volume threshold
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            # If no valid mid-market price, use last stored price or default to midpoint
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # Apply mean-reversion adjustment if last price is known
            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            
            # Store the last computed price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        """
        Places market orders to take the best available prices within a given width.
        
        - Avoids adverse selection if enabled.
        - Takes into account position limits.
        """
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        """
        Places orders to clear existing positions at a fair price within a given width.
        
        - Ensures that the trader does not hold excessive inventory.
        - Helps maintain liquidity by adjusting to market conditions.
        """
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    
    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # Ignore trades within this edge for pennying or joining
        join_edge: float,  # Join trades within this edge
        default_edge: float,  # Default edge to request if there are no levels to penny or join
        manage_position: bool = False,  # Whether to adjust bids/asks based on position
        soft_position_limit: int = 0,  # Threshold to modify orders based on position
    ):
        orders: List[Order] = []
        
        # Identify ask prices above the fair value, considering the disregard edge
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        
        # Identify bid prices below the fair value, considering the disregard edge
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        # Determine the best ask and best bid beyond the fair value range
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        
        # Set default bid and ask prices based on fair value and default edge
        ask = round(fair_value + default_edge)
        bid = round(fair_value - default_edge)
        
        # Adjust ask price if there's an order close enough to fair value
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # Join the existing order
            else:
                ask = best_ask_above_fair - 1  # Undercut by a small amount (pennying)
        
        # Adjust bid price if there's an order close enough to fair value
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # Join the existing order
            else:
                bid = best_bid_below_fair + 1  # Outbid slightly
        
        # Modify orders if position management is enabled
        if manage_position:
            if position > soft_position_limit:
                ask -= 1  # Reduce ask price to sell faster
            elif position < -1 * soft_position_limit:
                bid += 1  # Increase bid price to buy faster
        
        # Execute market making strategy
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        
        return orders, buy_order_volume, sell_order_volume
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        
        # Process trading for RAINFOREST_RESIN
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            # Execute take, clear, and make orders for RAINFOREST_RESIN
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )
        
        # Process trading for KELP
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )
        
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        
        return result, conversions, traderData
