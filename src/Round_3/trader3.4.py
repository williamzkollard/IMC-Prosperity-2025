from typing import List, Dict, Tuple, Optional, Any
import string
import jsonpickle
import numpy as np
import math
from collections import deque
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from math import log, sqrt, exp
from statistics import NormalDist

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


        

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


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
        "take_width": 2, #1
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.18, #-0.48
        "disregard_edge": 2, #1
        "join_edge": 0, #1
        "default_edge": 1, #2
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
        "optimal_z": 5, #optimised
        "scaling_pct": 0.01 #optimised / add desired volume
    },
    Product.SPREAD1: {
        "default_spread_mean": 48.777856,
        "default_spread_std": 85.119723,
        "spread_window": 55,
        "zscore_threshold": 3,
        "target_position": 100,
        "exit_threshold": 0
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2336,
        "default_spread_std": 59.8536,
        "spread_window": 60,
        "zscore_threshold": 5,
        "target_position": 100,
        "exit_threshold": 0
    },
    

    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.152193, #calculate from data
        "threshold": 0.127426, #unsure? 
        "strike": 10000,  #it is strange but works better
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 21, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.183454 , #calculate from data
        "threshold": 0.062167, #unsure? 
        "strike": 9750,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 21, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.174124, #calculate from data
        "threshold": 0.007581, #unsure? 
        "strike": 10000,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 21, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.158651, #calculate from data
        "threshold": 0.004627, #unsure? 
        "strike": 10250,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 25, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.154997, #calculate from data
        "threshold": 0.005499, #unsure? 
        "strike": 10500,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 25, #calculate from data
    },
}

BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}



class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)


    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0 #i.e. 100 vol
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility



class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
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
    #                     ORDER & POSITION METHODS for KELP & RESIN
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
    #                     METHODS FOR VOLCANIC_ROCK ORDERS
    ########################################################################
    
    def get_volcanic_rock_voucher_mid_price(
        self, voucher_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(voucher_order_depth.buy_orders) > 0
            and len(voucher_order_depth.sell_orders) > 0
        ):
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            traderData["prev_voucher_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_voucher_price"]
        

    def delta_hedge_volcanic_rock_position(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_position: int,
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the overall position in COCONUT_COUPON by creating orders in COCONUT.

        Args:
            coconut_order_depth (OrderDepth): The order depth for the COCONUT product.
            coconut_coupon_position (int): The current position in COCONUT_COUPON.
            coconut_position (int): The current position in COCONUT.
            coconut_buy_orders (int): The total quantity of buy orders for COCONUT in the current iteration.
            coconut_sell_orders (int): The total quantity of sell orders for COCONUT in the current iteration.
            delta (float): The current value of delta for the COCONUT_COUPON product.
            traderData (Dict[str, Any]): The trader data for the COCONUT_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the COCONUT_COUPON position.
        """

        target_volcanic_rock_position = -int(delta * volcanic_rock_voucher_position)
        hedge_quantity = volcanic_rock_position - (
            volcanic_rock_position + volcanic_rock_buy_orders - volcanic_rock_sell_orders
        )

        orders: List[Order] = []
        if hedge_quantity > 0:
            # Buy volcanic_rock
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(hedge_quantity), -volcanic_rock_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.COCONUT] - (volcanic_rock_position + volcanic_rock_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, quantity))
        elif hedge_quantity < 0:
            # Sell volcanic_rock
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(hedge_quantity), volcanic_rock_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] + (volcanic_rock_position - volcanic_rock_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders    
        
    def delta_hedge_volcanic_rock_voucher_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the new orders for COCONUT_COUPON by creating orders in COCONUT.

        Args:
            coconut_order_depth (OrderDepth): The order depth for the COCONUT product.
            coconut_coupon_orders (List[Order]): The new orders for COCONUT_COUPON.
            coconut_position (int): The current position in COCONUT.
            coconut_buy_orders (int): The total quantity of buy orders for COCONUT in the current iteration.
            coconut_sell_orders (int): The total quantity of sell orders for COCONUT in the current iteration.
            delta (float): The current value of delta for the COCONUT_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the new COCONUT_COUPON orders.
        """
        if len(volcanic_rock_voucher_orders) == 0:
            return None

        net_volcanic_rock_voucher_quantity = sum(
            order.quantity for order in volcanic_rock_voucher_orders
        )
        target_volcanic_rock_quantity = -int(delta * net_volcanic_rock_voucher_quantity)

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy COCONUT
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity), -volcanic_rock_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] - (volcanic_rock_position + volcanic_rock_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif target_volcanic_rock_quantity < 0:
            # Sell COCONUT
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity), volcanic_rock_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] + (volcanic_rock_position - volcanic_rock_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders

        
    def volcanic_rock_voucher_orders(
        self,
        product: str,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        traderData: Dict[str, Any],
        zscore: float,
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:

        traderData["past_voucher_vol"].append(zscore)
        if len(traderData["past_voucher_vol"]) < self.params[product]["std_window"]:
            return None, None
        if len(traderData["past_voucher_vol"]) > self.params[product]["std_window"]:
            traderData["past_voucher_vol"].pop(0)

        if zscore >= self.params[product]["zscore_threshold"]:
            if voucher_position != -self.LIMIT[product]:
                target_voucher_position = -self.LIMIT[product]
                if len(voucher_order_depth.buy_orders) > 0:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(target_voucher_position - voucher_position)
                    quantity = min(target_quantity, abs(voucher_order_depth.buy_orders[best_bid]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_bid, -quantity)], []
                    else:
                        return [Order(product, best_bid, -quantity)], [Order(product, best_bid, -quote_quantity)]

        elif zscore <= -self.params[product]["zscore_threshold"]:
            if voucher_position != self.LIMIT[product]:
                target_voucher_position = self.LIMIT[product]
                if len(voucher_order_depth.sell_orders) > 0:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(target_voucher_position - voucher_position)
                    quantity = min(target_quantity, abs(voucher_order_depth.sell_orders[best_ask]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_ask, quantity)], []
                    else:
                        return [Order(product, best_ask, quantity)], [Order(product, best_ask, quote_quantity)]

        return None, None



    def volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_position: int,
        delta: float) -> List[Order]:

        if volcanic_rock_voucher_orders == None or len(volcanic_rock_voucher_orders) == 0:
            volcanic_rock_voucher_position_after_trade = volcanic_rock_voucher_position
        else:
            volcanic_rock_voucher_position_after_trade = volcanic_rock_voucher_position + sum(
                order.quantity for order in volcanic_rock_voucher_orders
            )

        target_volcanic_rock_position = -delta * volcanic_rock_voucher_position_after_trade

        if target_volcanic_rock_position == volcanic_rock_position:
            return None

        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, round(quantity)))

        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -round(quantity)))

        return orders


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
        #                VOLCANIC ROCK ORDERS
   ####################################################################
        """
        volcanic_vouchers = [
        Product.VOLCANIC_ROCK_VOUCHER_9500,
        Product.VOLCANIC_ROCK_VOUCHER_9750,
        Product.VOLCANIC_ROCK_VOUCHER_10000,
        Product.VOLCANIC_ROCK_VOUCHER_10250,
        Product.VOLCANIC_ROCK_VOUCHER_10500,
        ]
        """

        volcanic_vouchers = []

        for voucher in volcanic_vouchers:
            if voucher not in traderObject:
                traderObject[voucher] = {
                    "prev_voucher_price": 0,
                    "past_voucher_vol": [],
                    "strike_iv_data": {},
                }

            if "base_iv_data" not in traderObject:
                traderObject["base_iv_data"] = []

            # Check if it's tradable and has order depth
            if voucher in self.params and voucher in state.order_depths:
                voucher_position = state.position.get(voucher, 0)
                volcanic_position = state.position.get(Product.VOLCANIC_ROCK, 0)

                volcanic_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
                voucher_order_depth = state.order_depths[voucher]

                volcanic_mid_price = (
                    max(volcanic_order_depth.buy_orders.keys())
                    + min(volcanic_order_depth.sell_orders.keys())
                ) / 2
                voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(  
                    voucher_order_depth,
                    traderObject[voucher],
                )

                tte = (
                    self.params[voucher]["starting_time_to_expiry"]
                    - (state.timestamp) / 1000000 / 365
                )

                
                iv = BlackScholes.implied_volatility(
                    voucher_mid_price,
                    volcanic_mid_price,
                    self.params[voucher]["strike"],
                    tte,
                )

                
                traderObject["base_iv_data"].append(iv)
                if len(traderObject["base_iv_data"]) > 50:
                    traderObject["base_iv_data"].pop(0)
                base_iv = np.mean(traderObject["base_iv_data"])

                
                strike = self.params[voucher]["strike"]
                traderObject[voucher]["strike_iv_data"][strike] = iv

                
                strikes = list(traderObject[voucher]["strike_iv_data"].keys())
                ivs = list(traderObject[voucher]["strike_iv_data"].values())

                fitted_iv = iv
                if len(strikes) >= 3:
                    coeffs = np.polyfit(strikes, ivs, 2)
                    fitted_iv = np.polyval(coeffs, strike)

                
                smile_z = (iv - fitted_iv) / (np.std(ivs) + 1e-6)


                delta = BlackScholes.delta(
                    volcanic_mid_price,
                    self.params[voucher]["strike"],
                    tte,
                    base_iv,
                )

                take_orders, make_orders = self.volcanic_rock_voucher_orders(
                    voucher,
                    voucher_order_depth,
                    voucher_position,
                    traderObject[voucher],
                    smile_z,
                )

                hedge_orders = self.volcanic_rock_hedge_orders(
                    volcanic_order_depth,
                    take_orders,
                    volcanic_position,
                    voucher_position,
                    delta,
                )

                if take_orders or make_orders:
                    result[voucher] = take_orders + make_orders

                if hedge_orders:
                    result[Product.VOLCANIC_ROCK] = hedge_orders


        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
