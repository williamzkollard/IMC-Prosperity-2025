from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math
from collections import deque

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 50,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.48,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 2,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "reversion_beta": -0.18,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "arima_window": 15,        # Larger window for AR(2)-like approach
        "price_momentum_factor": 0.2,
        "volatility_threshold": 1.5,
        "slope_threshold": 0.3
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }

        # Initialize price history for each product
        self.price_history = {
            Product.RAINFOREST_RESIN: deque(maxlen=50),
            Product.KELP: deque(maxlen=50),
            # Make a bigger buffer for SQUID_INK so AR(2)-like is more stable
            Product.SQUID_INK: deque(maxlen=200)
        }

    ########################################################################
    #                             HELPER METHODS
    ########################################################################
    def take_best_orders(
        self,
        product: str,
        fair_value: float,
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
        'Take' logic: if best ask is meaningfully below fair_value, buy it,
        if best bid is meaningfully above fair_value, sell it.
        """
        position_limit = self.LIMIT[product]

        # Attempt to buy from the best ask if cheap
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            # Check 'prevent_adverse' to avoid huge lumps
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        # Adjust local order book so we don't double-take
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Attempt to sell to the best bid if rich
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )
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
        'Make' logic: place our own passive bid & ask around fair_value.
        """
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

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
    ) -> (int, int):
        """
        'Clear' logic: if we have a net positive position and there's
        a decent buy order above fair_value, let's lighten up. 
        Similarly for a net negative position and a cheap ask below fair_value.
        """
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # If net positive > 0, see if we can sell some at or above fair_value
        if position_after_take > 0:
            # Sum volume from all buy orders with price >= fair_for_ask
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

        # If net negative < 0, see if we can buy some at or below fair_value
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_order_volume, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid price from best ask and best bid."""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

    ########################################################################
    #                        KELP FAIR VALUE (SAME)
    ########################################################################
    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # Mean reversion step
            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    ########################################################################
    #                     SQUID INK FAIR VALUE (IMPROVED)
    ########################################################################
    
    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject, market_trades=None) -> float:
        """
        New & improved AR(2)-like approach plus momentum for SQUID_INK.
          - We'll keep storing mid_prices into traderObject["squid_ink_prices"].
          - Then compute average 1st difference and 2nd difference for an AR(2) guess.
          - We'll also compute a short rolling slope for momentum.
        """
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is None:
            return None

        # Ensure we have a list of historical prices in the dict
        if "squid_ink_prices" not in traderObject:
            traderObject["squid_ink_prices"] = []
        traderObject["squid_ink_prices"].append(mid_price)

        # If we exceed the window, trim from the front
        if len(traderObject["squid_ink_prices"]) > self.params[Product.SQUID_INK]["arima_window"]:
            traderObject["squid_ink_prices"].pop(0)

        prices = traderObject["squid_ink_prices"]
        n = len(prices)
        if n < 5:
            # Not enough data for AR(2)-like. Just return the mid_price as fair.
            traderObject["squid_ink_last_price"] = mid_price
            return mid_price
        
        # === MODE DETECTION ===
        # # Rolling volatility and slope
        recent_returns = np.diff(prices[-5:])
        volatility = np.std(recent_returns)
        slope = (prices[-1] - prices[-5]) / 4  # ~linear trend

        # Save into traderObject for access elsewhere if needed
        traderObject["volatility"] = volatility
        traderObject["slope"] = slope

        # Simple regime classification
        vol_thresh = self.params[Product.SQUID_INK].get("volatility_threshold", 1)
        slope_thresh = self.params[Product.SQUID_INK].get("slope_threshold", 0.2)
    
        if volatility > vol_thresh:
            regime = "volatile"
        elif abs(slope) > slope_thresh:
            regime = "trending"
        else:
            regime = "sideways"

        traderObject["regime"] = regime

        # 1) Mean Reversion from last trade
        reversion_component = 0
        if traderObject.get("squid_ink_last_price") is not None:
            last_price = traderObject["squid_ink_last_price"]
            last_returns = (mid_price - last_price) / last_price
            reversion_component = last_returns * self.params[Product.SQUID_INK]["reversion_beta"]

        # 2) AR(2)-like difference approach
        #    avg_diff ~ average of (p[i] - p[i-1])
        #    avg_diff2 ~ average of ( (p[i]-p[i-1]) - (p[i-1]-p[i-2]) )
        #    Next price guess = p[n-1] + avg_diff + 0.5*avg_diff2
        diffs = []
        for i in range(1, n):
            diffs.append(prices[i] - prices[i-1])
        avg_diff = np.mean(diffs) if len(diffs) > 0 else 0

        diff2 = []
        for i in range(1, len(diffs)):
            diff2.append(diffs[i] - diffs[i-1])
        avg_diff2 = np.mean(diff2) if len(diff2) > 0 else 0

        arima_component = avg_diff + 0.5 * avg_diff2

        # 3) Momentum using a rolling slope on the last ~5 prices
        momentum_component = 0
        lookback = min(5, n)  # e.g. take last 5
        recent_slice = prices[-lookback:]
        if len(recent_slice) > 1:
            # slope approx = (p[end] - p[start]) / (lookback - 1)
            slope = (recent_slice[-1] - recent_slice[0]) / (lookback - 1)
            # scale slope by some factor
            momentum_component = slope * self.params[Product.SQUID_INK]["price_momentum_factor"]

        # Combine them:
        # === Combine dynamically based on regime ===
        if regime == "sideways":
            fair_value = mid_price * (1 + reversion_component)
        elif regime == "trending":
            fair_value = mid_price + arima_component + momentum_component
        else:  # volatile
            fair_value = mid_price  # stay close to market to avoid risk

        traderObject["squid_ink_last_price"] = mid_price
        return fair_value
    

    ########################################################################
    #                     ORDER & POSITION METHODS
    ########################################################################
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
        """Wrapper that calls 'take_best_orders' and returns orders + volumes."""
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
        """Wrapper that calls 'clear_position_order' and returns orders + volumes."""
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
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        """
        'Make' logic: figure out which price levels to join or slightly improve,
        then place buy/sell if within position constraints.
        """
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # If user wants to tilt based on position
        if manage_position:
            if position > soft_position_limit:
                # If we are heavily long, shift our ask slightly cheaper to reduce position
                ask = max(ask - 1, 1)
            elif position < -soft_position_limit:
                # If we are heavily short, shift our bid up slightly
                bid += 1

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

    def update_price_history(self, product, mid_price, traderObject):
        """Keep track of mid_price in traderObject, separate from built-in 'squid_ink_prices'."""
        key = f"{product.lower()}_price_history"
        if key not in traderObject:
            traderObject[key] = []
        traderObject[key].append(mid_price)
        # Keep only most recent 50
        if len(traderObject[key]) > 50:
            traderObject[key].pop(0)

    ########################################################################
    #                              RUN METHOD
    ########################################################################
    def run(self, state: TradingState):
        # Re-hydrate the dictionary containing our stored state
        traderObject = {}
        if state.traderData:
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except:
                traderObject = {}

        result = {}

        ####################################################################
        #                RAINFOREST_RESIN (kept identical)
        ####################################################################
        if (Product.RAINFOREST_RESIN in self.params 
            and Product.RAINFOREST_RESIN in state.order_depths):
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
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

        ####################################################################
        #                KELP (kept identical)
        ####################################################################
        if (Product.KELP in self.params 
            and Product.KELP in state.order_depths):
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            if kelp_fair_value is not None:
                kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
                kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
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

        ####################################################################
        #                SQUID_INK (IMPROVED)
        ####################################################################
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)
            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK],
                traderObject,
                state.market_trades.get(Product.SQUID_INK, [])
            )

        if squid_ink_fair_value is not None:
            regime = traderObject.get("regime", "sideways")
            buy_order_volume, sell_order_volume = 0, 0
            squid_ink_orders = []

            if regime == "trending":
                # Only aggressive take orders to follow the trend
                squid_ink_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    self.params[Product.SQUID_INK]["take_width"] * 1.2,  # slightly wider
                    squid_ink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"]
                )

            elif regime == "sideways":
                # Market making + some opportunity taking
                take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    squid_ink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"]
                )
                clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume
                )
                make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.SQUID_INK]["disregard_edge"],
                    self.params[Product.SQUID_INK]["join_edge"],
                    self.params[Product.SQUID_INK]["default_edge"]
                )
                squid_ink_orders = take_orders + clear_orders + make_orders

            elif regime == "volatile":
                # Only clear bad positions
                squid_ink_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"] * 1.5,  # more conservative
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume
                )

            result[Product.SQUID_INK] = squid_ink_orders
        ####################################################################
        #       Update price history for logs (optional debugging)
        ####################################################################
        for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            if product in state.order_depths:
                mid_price = self.calculate_mid_price(state.order_depths[product])
                if mid_price is not None:
                    self.update_price_history(product, mid_price, traderObject)

        ####################################################################
        #       Final Return
        ####################################################################
        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
