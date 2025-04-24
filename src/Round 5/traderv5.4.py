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
import statistics 


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
        "mean_volatility": 0.0333699, #calculate from data
        "threshold": 0.06, #unsure? 
        "strike": 9500,
        "starting_time_to_expiry": 5/7, #recompute each round
        "std_window": 25, #calculate from data?
        "diff_threshold": 0.001, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.03442, #calculate from data
        "threshold": 0.06, #unsure? 
        "strike": 9750,
        "starting_time_to_expiry": 5/7, #recompute each round - end of round 3 is 4 days 
        "std_window": 10, #calculate from data?
        "diff_threshold": 0.001, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.0312, #calculate from data
        "threshold": 0.00163, #unsure? 
        "strike": 10000,
        "starting_time_to_expiry": 5/7, #recompute each round
        "std_window": 6, #calculate from data?
        "diff_threshold": 0.001, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.029347, #calculate from data
        "threshold": 0.00163, #unsure? 
        "strike": 10250,
        "starting_time_to_expiry": 5/7, #recompute each round
        "std_window": 6, #calculate from data?
        "diff_threshold": 0.001, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.0300179, #calculate from data
        "threshold": 0.00163, #unsure? 
        "strike": 10500,
        "starting_time_to_expiry": 5/7, #recompute each round
        "std_window": 2.5, #calculate from data?
        "diff_threshold": 0.001, #calculate from data
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
        d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * statistics.NormalDist().cdf(d1) - strike * statistics.NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        return statistics.NormalDist().cdf(d1)


    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        # binary search ts
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
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
    

    def make_squid_ink_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        traderObject: dict,
    ) -> List[Order]:
        orders: List[Order] = []

        max_position = self.LIMIT[Product.SQUID_INK]
        signal = traderObject.get("squid_ink_signal", "NEUTRAL")
        z_score = traderObject.get("squid_ink_zscore", 0)
        default_edge = self.params[Product.SQUID_INK]["default_edge"]
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        bid = round(fair_value - default_edge)
        ask = round(fair_value + default_edge)
        mid_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else fair_value
        scaling_pct = self.params[Product.SQUID_INK]["scaling_pct"]
        price_adjustment = int(z_score * mid_price * scaling_pct)

        if signal == "BUY":
            bid += abs(price_adjustment)
        elif signal == "SELL":
            ask -= abs(price_adjustment)

        if position > 25:
            ask = max(ask - 1, 1)
        elif position < -25:
            bid += 1

        volume = 5  # fixed volume for simplicity

        if position + volume <= max_position:
            orders.append(Order(Product.SQUID_INK, bid, volume))
        if position - volume >= -max_position:
            orders.append(Order(Product.SQUID_INK, ask, -volume))

        return orders
    

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
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is None:
            return None

        # Save mid price into rolling history
        if "squid_ink_prices" not in traderObject:
            traderObject["squid_ink_prices"] = []
        traderObject["squid_ink_prices"].append(mid_price)

        # Keep history bounded for both ARIMA and Z-score logic
        HISTORY_SIZE = max(60, self.params[Product.SQUID_INK]["arima_window"])
        if len(traderObject["squid_ink_prices"]) > HISTORY_SIZE:
            traderObject["squid_ink_prices"].pop(0)

        prices = traderObject["squid_ink_prices"]
        n = len(prices)
        if n < 5:
            traderObject["squid_ink_last_price"] = mid_price
            return mid_price

        # === Z-score and signal ===
        rolling_window = 30
        if len(prices) >= rolling_window:
            recent_prices = prices[-rolling_window:]
            mean = np.mean(recent_prices)
            std = np.std(recent_prices)

            z_score = (mid_price - mean) / std if std > 0 else 0.0
            traderObject["squid_ink_zscore"] = z_score

            if z_score > self.params[Product.SQUID_INK]["optimal_z"]:
                traderObject["squid_ink_signal"] = "SELL"
            elif z_score < -self.params[Product.SQUID_INK]["optimal_z"]:
                traderObject["squid_ink_signal"] = "BUY"
            else:
                traderObject["squid_ink_signal"] = "NEUTRAL"

        # Mean Reversion Component 
        reversion_component = 0
        if traderObject.get("squid_ink_last_price") is not None:
            last_price = traderObject["squid_ink_last_price"]
            last_returns = (mid_price - last_price) / last_price
            reversion_component = last_returns * self.params[Product.SQUID_INK]["reversion_beta"]

        # ARIMA(2)-like Component 
        diffs = [prices[i] - prices[i - 1] for i in range(1, n)]
        avg_diff = np.mean(diffs) if diffs else 0

        diff2 = [diffs[i] - diffs[i - 1] for i in range(1, len(diffs))]
        avg_diff2 = np.mean(diff2) if diff2 else 0

        arima_component = avg_diff + 0.5 * avg_diff2

        # === Momentum Component ===
        momentum_component = 0
        recent_slice = prices[-min(5, n):]
        if len(recent_slice) > 1:
            slope = (recent_slice[-1] - recent_slice[0]) / (len(recent_slice) - 1)
            momentum_component = slope * self.params[Product.SQUID_INK]["price_momentum_factor"]

        # Final unified fair value
        fair_value = mid_price + arima_component + momentum_component + (mid_price * reversion_component)

        traderObject["squid_ink_last_price"] = mid_price
        return fair_value



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
        

    def volcanic_rock_vol_spread_orders(
        self,
        overpriced_voucher: str,
        underpriced_voucher: str,
        order_depths: Dict[str, OrderDepth],
        positions: Dict[str, int],
        traderObject: Dict[str, Any],
    ) -> Tuple[List[Order], List[Order]]:
        """
        Executes a volatility spread trade: Sell overpriced option, buy underpriced one.
        """
        take_orders = []
        make_orders = []

        # Overpriced leg
        pos1 = positions.get(overpriced_voucher, 0)
        od1 = order_depths[overpriced_voucher]
        limit1 = self.LIMIT[overpriced_voucher]
        max_qty1 = abs(-limit1 - pos1)

        if len(od1.buy_orders) > 0:
            best_bid = max(od1.buy_orders.keys())
            qty1 = min(max_qty1, abs(od1.buy_orders[best_bid]))
            quote_qty1 = max_qty1 - qty1

            if qty1 > 0:
                take_orders.append(Order(overpriced_voucher, best_bid, -qty1))
            if quote_qty1 > 0:
                make_orders.append(Order(overpriced_voucher, best_bid, -quote_qty1))

        # Underpriced leg
        pos2 = positions.get(underpriced_voucher, 0)
        od2 = order_depths[underpriced_voucher]
        limit2 = self.LIMIT[underpriced_voucher]
        max_qty2 = abs(limit2 - pos2)

        if len(od2.sell_orders) > 0:
            best_ask = min(od2.sell_orders.keys())
            qty2 = min(max_qty2, abs(od2.sell_orders[best_ask]))
            quote_qty2 = max_qty2 - qty2

            if qty2 > 0:
                take_orders.append(Order(underpriced_voucher, best_ask, qty2))
            if quote_qty2 > 0:
                make_orders.append(Order(underpriced_voucher, best_ask, qty2))

        return take_orders, make_orders


        

    def volcanic_rock_delta_hedge_orders(
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

        volcanic_vouchers = [
    Product.VOLCANIC_ROCK_VOUCHER_9500,
    Product.VOLCANIC_ROCK_VOUCHER_9750,
    Product.VOLCANIC_ROCK_VOUCHER_10000,
    Product.VOLCANIC_ROCK_VOUCHER_10250,
    Product.VOLCANIC_ROCK_VOUCHER_10500,
]

        # Initialize data structures
        for voucher in volcanic_vouchers:
            if voucher not in traderObject:
                traderObject[voucher] = {
                    "prev_voucher_price": 0,
                    "past_voucher_vol": [],
                }

        # Step 1: Base IV tracking
        if "base_iv_data" not in traderObject:
            traderObject["base_iv_data"] = []

        # Step 2: Gather strike-to-IV data
        strike_iv_data = {}
        for voucher in volcanic_vouchers:
            if voucher in self.params and voucher in state.order_depths:
                voucher_order_depth = state.order_depths[voucher]
                volcanic_order_depth = state.order_depths[Product.VOLCANIC_ROCK]

                voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(
                    voucher_order_depth,
                    traderObject[voucher],
                )

                volcanic_mid_price = (
                    max(volcanic_order_depth.buy_orders.keys())
                    + min(volcanic_order_depth.sell_orders.keys())
                ) / 2

                strike = self.params[voucher]["strike"]
                tte = (
                    self.params[voucher]["starting_time_to_expiry"]
                    - (state.timestamp) / 1000000 / 365
                )

                iv = BlackScholes.implied_volatility(
                    voucher_mid_price,
                    volcanic_mid_price,
                    strike,
                    tte,
                )

                strike_iv_data[strike] = iv

        # Step 3: Fit the volatility smile
        fitted_coeffs = None
        base_iv = None
        if len(strike_iv_data) >= 3:
            strikes = list(strike_iv_data.keys())
            ivs = list(strike_iv_data.values())
            fitted_coeffs = np.polyfit(strikes, ivs, 2)
            base_iv = np.polyval(fitted_coeffs, volcanic_mid_price)
            traderObject["base_iv_data"].append(base_iv)
            if len(traderObject["base_iv_data"]) > 50:
                traderObject["base_iv_data"].pop(0)
            base_iv = np.mean(traderObject["base_iv_data"])

        # Step 4: Detect mispricings
        entry_threshold = 0.001
        most_overpriced = None
        most_underpriced = None

        for voucher in volcanic_vouchers:
            if voucher in self.params and voucher in state.order_depths:
                strike = self.params[voucher]["strike"]
                tte = (
                    self.params[voucher]["starting_time_to_expiry"]
                    - (state.timestamp) / 1000000 / 365
                )
                iv = strike_iv_data.get(strike)
                if iv is None or fitted_coeffs is None:
                    continue

                fitted_iv = np.polyval(fitted_coeffs, strike)
                abs_diff = iv - fitted_iv

                if abs_diff > entry_threshold:
                    if most_overpriced is None or abs_diff > most_overpriced["diff"]:
                        most_overpriced = {
                            "voucher": voucher,
                            "diff": abs_diff,
                            "strike": strike,
                            "iv": iv,
                        }

                elif abs_diff < -entry_threshold:
                    if most_underpriced is None or abs_diff < most_underpriced["diff"]:
                        most_underpriced = {
                            "voucher": voucher,
                            "diff": abs_diff,
                            "strike": strike,
                            "iv": iv,
                        }

        # Step 5: Execute spread and hedge
        if most_overpriced and most_underpriced:
            take_orders, make_orders = self.volcanic_rock_vol_spread_orders(
                most_overpriced["voucher"],
                most_underpriced["voucher"],
                state.order_depths,
                state.position,
                traderObject,
            )

            if take_orders or make_orders:
                all_orders = take_orders + make_orders

                result[most_overpriced["voucher"]] = [
                    o for o in all_orders if o.symbol == most_overpriced["voucher"]
                ]
                result[most_underpriced["voucher"]] = [
                    o for o in all_orders if o.symbol == most_underpriced["voucher"]
                ]

                traderObject["open_spread_trade"] = {
                    "over": most_overpriced["voucher"],
                    "under": most_underpriced["voucher"],
                    "entry_diff": most_overpriced["diff"] - most_underpriced["diff"],
                    "timestamp": state.timestamp,
                }

                
                # Step 6: Hedge the spread's net delta
                volcanic_position = state.position.get(Product.VOLCANIC_ROCK, 0)

                strike_over = most_overpriced["strike"]
                tte_over = self.params[most_overpriced["voucher"]]["starting_time_to_expiry"] - (state.timestamp / 1_000_000 / 365)

                strike_under = most_underpriced["strike"]
                tte_under = self.params[most_underpriced["voucher"]]["starting_time_to_expiry"] - (state.timestamp / 1_000_000 / 365)

                delta_over = BlackScholes.delta(volcanic_mid_price, strike_over, tte_over, base_iv)
                delta_under = BlackScholes.delta(volcanic_mid_price, strike_under, tte_under, base_iv)
                net_delta = (-delta_over) + (delta_under)

                hedge_orders = self.volcanic_rock_delta_hedge_orders(
                    state.order_depths[Product.VOLCANIC_ROCK],
                    take_orders,
                    volcanic_position,
                    0,  # optional placeholder; not used in net-delta hedging
                    net_delta,
                )

                if hedge_orders:
                    result[Product.VOLCANIC_ROCK] = hedge_orders

                


        ####################################################################
        #                RAINFOREST_RESIN ORDERS
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
        #                KELP ORDERS
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
        #                SQUID_INK ORDERS
        ####################################################################
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)

            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK],
                traderObject,
                state.market_trades.get(Product.SQUID_INK, [])
            )

            if squid_ink_fair_value is not None:
                # Use custom quoting logic with signal & z-score scaling
                squid_ink_orders = self.make_squid_ink_orders(
                    order_depth=state.order_depths[Product.SQUID_INK],
                    fair_value=squid_ink_fair_value,
                    position=squid_ink_position,
                    traderObject=traderObject
                )

                result[Product.SQUID_INK] = squid_ink_orders


        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
