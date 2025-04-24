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
        "mean_volatility": 0.152193, #calculate from data
        "threshold": 0.1, #unsure? 
        "strike": 9500,  #it is strange but works better
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 2.5, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.183454 , #calculate from data
        "threshold": 0.1, #unsure? 
        "strike": 9750,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 2.5, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.174124, #calculate from data
        "threshold": 0.01, #unsure? 
        "strike": 10000,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 2.5, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.158651, #calculate from data
        "threshold": 0.728011, #unsure? 
        "strike": 10250,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 2.5, #calculate from data
    },

    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.154997, #calculate from data
        "threshold": 0.0552, #unsure? 
        "strike": 10500,
        "starting_time_to_expiry": 5/365, #recompute each round
        "std_window": 6, #calculate from data?
        "zscore_threshold": 2.5, #calculate from data
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
    #                     METHODS FOR BASKETS
    ########################################################################
    


    def get_microprice(self, order_depth):
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol +
                                                                      best_ask_vol)
    
    def artifical_order_depth(self, order_depths: Dict[str, OrderDepth],
                              picnic1: bool = True): 
        if picnic1:
            DJEMBES_PER_PICNIC = BASKET1_WEIGHTS[Product.DJEMBES]
            CROISSANT_PER_PICNIC = BASKET1_WEIGHTS[Product.CROISSANTS]
            JAM_PER_PICNIC = BASKET1_WEIGHTS[Product.JAMS]
            
        else:
            CROISSANT_PER_PICNIC = BASKET2_WEIGHTS[Product.CROISSANTS]
            JAM_PER_PICNIC = BASKET2_WEIGHTS[Product.JAMS]
            
        artifical_order_price = OrderDepth()
        
        croissant_best_bid = (max(order_depths[Product.CROISSANTS].buy_orders.keys()) 
                            if order_depths[Product.CROISSANTS].buy_orders
                            else 0)
            
        croissant_best_ask = (min(order_depths[Product.CROISSANTS].sell_orders.keys())
                            if order_depths[Product.CROISSANTS].sell_orders
                            else float("inf"))
        
        jams_best_bid = (max(order_depths[Product.JAMS].buy_orders.keys()) 
                            if order_depths[Product.JAMS].buy_orders
                            else 0)
        
        jams_best_ask = (min(order_depths[Product.JAMS].sell_orders.keys())
                            if order_depths[Product.JAMS].sell_orders
                            else float("inf"))
        
        if picnic1:
            djembes_best_bid = (max(order_depths[Product.DJEMBES].buy_orders.keys()) 
                                if order_depths[Product.DJEMBES].buy_orders
                                else 0)
                
            djembes_best_ask = (min(order_depths[Product.DJEMBES].sell_orders.keys())
                                if order_depths[Product.DJEMBES].sell_orders
                                else float("inf"))
            
            art_bid = (djembes_best_bid*DJEMBES_PER_PICNIC + 
                       croissant_best_bid*CROISSANT_PER_PICNIC +
                       jams_best_bid*JAM_PER_PICNIC)
            art_ask = (djembes_best_ask*DJEMBES_PER_PICNIC +
                       croissant_best_ask*CROISSANT_PER_PICNIC +
                       jams_best_ask*JAM_PER_PICNIC)
        else:
            art_bid = (croissant_best_bid * CROISSANT_PER_PICNIC + 
                       jams_best_bid * JAM_PER_PICNIC)
            art_ask = (croissant_best_ask * CROISSANT_PER_PICNIC +
                       jams_best_ask * JAM_PER_PICNIC)
            
        if art_bid > 0:
            croissant_bid_volume = (order_depths[Product.CROISSANTS].buy_orders[croissant_best_bid]
                // CROISSANT_PER_PICNIC)
            jams_bid_volume = (order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAM_PER_PICNIC)
            
            if picnic1:
                djembes_bid_volume = (order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                    // DJEMBES_PER_PICNIC)

                artifical_bid_volume = min(djembes_bid_volume, croissant_bid_volume, 
                                         jams_bid_volume)
            else:
                artifical_bid_volume = min(croissant_bid_volume, jams_bid_volume)
            artifical_order_price.buy_orders[art_bid] = artifical_bid_volume

        if art_ask < float("inf"):
            croissant_ask_volume = (-order_depths[Product.CROISSANTS].sell_orders[croissant_best_ask]
                // CROISSANT_PER_PICNIC)
            jams_ask_volume = (-order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAM_PER_PICNIC)
            
            if picnic1:
                djembes_ask_volume = (-order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                    // DJEMBES_PER_PICNIC)
                
                artifical_ask_volume = min(
                    djembes_ask_volume, croissant_ask_volume, jams_ask_volume
                )
            else:
                artifical_ask_volume = min(croissant_ask_volume, jams_ask_volume)
            artifical_order_price.sell_orders[art_ask] = -artifical_ask_volume

        return artifical_order_price
        
    def convert_orders(self, artifical_orders: List[Order],
                   order_depths: Dict[str, OrderDepth], state: TradingState,
                   picnic1: bool = True):
        
        if picnic1:
            component_weights = BASKET1_WEIGHTS
            component_orders = {
                Product.DJEMBES: [],
                Product.CROISSANTS: [],
                Product.JAMS: [],
            }
        else:
            component_weights = BASKET2_WEIGHTS
            component_orders = {
                Product.CROISSANTS: [],
                Product.JAMS: [],
            }

        for order in artifical_orders:
            quantity = order.quantity  # basket quantity

            # === Step 1: Calculate max safe basket quantity (component-limit aware) ===
            max_possible_qty = float('inf')
            for component, weight in component_weights.items():
                component_position = state.position.get(component, 0)
                component_limit = self.LIMIT[component]
                desired_qty = quantity * weight

                # Room left before hitting limit (based on direction of desired trade)
                if desired_qty > 0:
                    available = component_limit - component_position
                else:
                    available = component_limit + component_position

                if weight != 0:
                    max_qty_component_can_handle = available // abs(weight)
                    max_possible_qty = min(max_possible_qty, max_qty_component_can_handle)

            # === Step 2: Scale down basket trade if needed ===
            final_quantity = min(abs(quantity), abs(max_possible_qty))
            final_quantity *= (1 if quantity > 0 else -1)

            if final_quantity == 0:
                continue  # Can't safely trade this basket

            # === Step 3: Build hedge component orders ===
            for component, weight in component_weights.items():
                component_quantity = final_quantity * weight
                if component_quantity == 0:
                    continue

                # Pick appropriate price from order book
                if component_quantity > 0:
                    price = min(order_depths[component].sell_orders.keys())
                else:
                    price = max(order_depths[component].buy_orders.keys())

                component_orders[component].append(Order(component, price, component_quantity))

        return component_orders

    
    def execute_spreads(self, target_position: int,
                        picnic_position: int,
                        order_depths: Dict[str, OrderDepth],
                        state: TradingState,
                        picnic1: bool = True):
        if target_position == picnic_position:
            return None
        
        target_quantity = abs(target_position - picnic_position)
        picnic_order_depth = (order_depths[Product.PICNIC_BASKET1] if picnic1
                              else order_depths[Product.PICNIC_BASKET2])
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        
        if target_position > picnic_position:
            picnic_ask_price = min(picnic_order_depth.sell_orders.keys())
            picnic_ask_vol = abs(picnic_order_depth.sell_orders[picnic_ask_price])
            artifical_bid_price = min(artifical_order_depth.buy_orders.keys())
            artifical_bid_vol = abs(artifical_order_depth.buy_orders[artifical_bid_price])
            
            orderbook_volume = min(picnic_ask_vol, artifical_bid_vol)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_product = Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2
            current_basket_position = state.position.get(basket_product, 0)
            basket_limit = self.LIMIT[basket_product]
            remaining_room = basket_limit - current_basket_position
            execute_volume = min(execute_volume, remaining_room)

            if execute_volume <= 0:
                return None


            picnic_orders = [
                (Order(Product.PICNIC_BASKET1, picnic_ask_price, execute_volume)
                 if picnic1
                 else Order(Product.PICNIC_BASKET2, picnic_ask_price, execute_volume))
            ]
            artifical_orders = [
                (Order(Product.SYNTHETIC1, artifical_bid_price, -execute_volume) # tbh does it matter if we used two artifical names
                 )
            ]

            aggregate_orders = self.convert_orders(
                artifical_orders, order_depths, state, picnic1
            )
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders
        else:
            picnic_bid_price = min(picnic_order_depth.buy_orders.keys())
            picnic_bid_vol = abs(picnic_order_depth.buy_orders[picnic_bid_price])
            artifical_ask_price = min(artifical_order_depth.sell_orders.keys())
            artifical_ask_vol = abs(artifical_order_depth.sell_orders[artifical_ask_price])
            
            orderbook_volume = min(picnic_bid_vol, artifical_ask_vol)
            execute_volume = min(orderbook_volume, target_quantity)


            basket_product = Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2
            current_basket_position = state.position.get(basket_product, 0)
            basket_limit = self.LIMIT[basket_product]
            remaining_room = basket_limit + current_basket_position  # since position is negative when shorting
            execute_volume = min(execute_volume, remaining_room)


            if execute_volume <= 0:
                return None


            picnic_orders = [
                (Order(Product.PICNIC_BASKET1, picnic_bid_price, -execute_volume)
                 if picnic1
                 else Order(Product.PICNIC_BASKET2, picnic_bid_price, -execute_volume))
            ]
            artifical_orders = [
                (Order(Product.SYNTHETIC1, artifical_ask_price, -execute_volume) 
                 )
            ]

            aggregate_orders = self.convert_orders(
                artifical_orders, order_depths, state, picnic1
            )
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth],
                      product: Product, picnic_position: int, 
                      spread_data: Dict[str, Any],
                      state: TradingState,
                      SPREAD,
                      picnic1: bool = True,
                      ):
        if (Product.PICNIC_BASKET1 not in order_depths.keys() or
            Product.PICNIC_BASKET2 not in order_depths.keys()):
            return None
        
        picnic_order_depth = (order_depths[Product.PICNIC_BASKET1] if picnic1
                              else order_depths[Product.PICNIC_BASKET2])
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        picnic_mprice = self.get_microprice(picnic_order_depth)
        artifical_mprice = self.get_microprice(artifical_order_depth)
        spread = picnic_mprice - artifical_mprice
        spread_data["spread_history"].append(spread)
        
        if (len(spread_data["spread_history"])
            < self.params[SPREAD]["spread_window"]):
            return None
        elif len(spread_data["spread_history"]) > self.params[SPREAD]["spread_window"]:
            spread_data["spread_history"].pop(0)
        
        spread_std = np.std(spread_data["spread_history"])
        
        zscore = (spread - self.params[SPREAD]["default_spread_mean"]) / spread_std
        
        if zscore >= self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != -self.params[SPREAD]["target_position"]:
                spread_data["entry"] = {
                "picnic_price": picnic_mprice,
                "synthetic_price": artifical_mprice,
                "position": -self.params[SPREAD]["target_position"]
            }
                return self.execute_spreads(
                    -self.params[SPREAD]["target_position"],
                    picnic_position,
                    order_depths,
                    state,
                    picnic1
                )
        
        if zscore <= -self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != self.params[SPREAD]["target_position"]:
                spread_data["entry"] = {
                "picnic_price": picnic_mprice,
                "synthetic_price": artifical_mprice,
                "position": self.params[SPREAD]["target_position"]
            }
                return self.execute_spreads(
                    self.params[SPREAD]["target_position"],
                    picnic_position,
                    order_depths,
                    state,
                    picnic1
                )

        # Exit logic: spread compressed AND it's profitable
        exit_threshold = self.params[SPREAD].get("exit_threshold", 0.5)
        entry = spread_data.get("entry", None)

        if abs(zscore) < exit_threshold and picnic_position != 0 and entry:
            entry_spread = entry["picnic_price"] - entry["synthetic_price"]
            current_spread = picnic_mprice - artifical_mprice

            # Long position → we want spread to grow
            if picnic_position > 0 and current_spread > entry_spread:
                spread_data["entry"] = None
                return self.execute_spreads(
                    0,
                    picnic_position,
                    order_depths,
                    state,
                    picnic1
                )

            # Short position → we want spread to shrink
            elif picnic_position < 0 and current_spread < entry_spread:
                spread_data["entry"] = None
                return self.execute_spreads(
                    0,
                    picnic_position,
                    order_depths,
                    state,
                    picnic1
                )

        spread_data["prev_zscore"] = zscore
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
        

        
    def volcanic_rock_voucher_orders(
        self,
        product: str,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:

   
        traderData["past_voucher_vol"].append(volatility)
        if (
            len(traderData["past_voucher_vol"])
            < self.params[product]["std_window"]
        ):
            return None, None

        elif (
            len(traderData["past_voucher_vol"])
            > self.params[product]["std_window"]
        ):
            traderData["past_voucher_vol"].pop(0)

        vol_z_score = (
            volatility - self.params[product]["mean_volatility"]
        ) / (np.std(traderData["past_voucher_vol"]) + 1e-8)

       
    

        if vol_z_score >= self.params[product]["zscore_threshold"]:
            if voucher_position != -self.LIMIT[product]:
                target_voucher_position = -self.LIMIT[product]
                if len(voucher_order_depth.buy_orders) > 0:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(
                        target_voucher_position - voucher_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(voucher_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_bid, -quantity)], []
                    else:
                        return [Order(product, best_bid, -quantity)], [
                            Order(product, best_bid, -quote_quantity)
                        ]

        elif vol_z_score <= -self.params[product]["zscore_threshold"]:
            if voucher_position != self.LIMIT[product]:
                target_voucher_position = self.LIMIT[product]
                if len(voucher_order_depth.sell_orders) > 0:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(
                        target_voucher_position - voucher_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(voucher_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_ask, quantity)], []
                    else:
                        return [Order(product, best_ask, quantity)], [
                            Order(product, best_ask, quote_quantity)
                        ]
        return None, None

    
    


    def volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_order_depth: OrderDepth,
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


   ####################################################################
        #                SPREAD 1 ORDERS
   ####################################################################

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        picnic1_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )

        
        spread1_orders = self.spread_orders(
                state.order_depths,
                Product.PICNIC_BASKET1,
                picnic1_position,
                traderObject[Product.SPREAD1],
                state,
                SPREAD = Product.SPREAD1,
                picnic1=True
            )
        if spread1_orders != None:
            result[Product.DJEMBES] = spread1_orders[Product.DJEMBES]
            result[Product.CROISSANTS] = spread1_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread1_orders[Product.JAMS]
            result[Product.PICNIC_BASKET1] = spread1_orders[Product.PICNIC_BASKET1]
        
        
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        
        
        picnic2_position = (state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position else 0)
        spread2_orders = self.spread_orders(
                state.order_depths,
                Product.PICNIC_BASKET2,
                picnic2_position,
                traderObject[Product.SPREAD2],
                state,
                SPREAD = Product.SPREAD2,
                picnic1=False
            )
        if spread2_orders != None:
            result[Product.CROISSANTS] = spread2_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread2_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]


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

        voucher_data = []
        
        volcanic_mid_price = self.get_volcanic_rock_voucher_mid_price(state.order_depths[Product.VOLCANIC_ROCK], traderObject)
        
        for voucher in volcanic_vouchers:
            if voucher in self.params and voucher in state.order_depths:
                voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(state.order_depths[voucher],
                                                                             traderObject.get(voucher, {}))
                K = self.params[voucher]["strike"]
                tte = self.params[voucher]["starting_time_to_expiry"]
                voucher_data.append({
                    'voucher': voucher,
                    'St': volcanic_mid_price,
                    'K': K,
                    'TTE': tte,
                    'Vt': voucher_mid_price
                })
        
        if len(voucher_data) >= 3:
            m_vals = []
            v_vals = []
            for item in voucher_data:
                m_val = np.log(item['K'] / item['St']) / np.sqrt(item['TTE']) #mt
                m_vals.append(m_val)
    
                v_val = BlackScholes.implied_volatility(item['Vt'], item['St'], item['K'], item['TTE'])
                v_vals.append(v_val)
            coeffs = np.polyfit(m_vals, v_vals, 2)
            base_iv = coeffs[2] 
            traderObject["voucher_regression_data"] = {"coeffs": coeffs, "base_iv": base_iv}
        
        for VOUCHER in volcanic_vouchers:
            if VOUCHER not in traderObject:
                traderObject[VOUCHER] = {
                    "prev_voucher_price": 0, 
                    "past_voucher_vol": [], 
                    }
                
            if VOUCHER in self.params and VOUCHER in state.order_depths:
                rock_voucher_position = state.position.get(VOUCHER, 0)
                rock_position = state.position.get(Product.VOLCANIC_ROCK, 0)
                rock_order_depth = state.order_depths.get(Product.VOLCANIC_ROCK, None)

                mid_price = self.get_volcanic_rock_voucher_mid_price(state.order_depths[VOUCHER], traderObject.get(VOUCHER, {}))
                K_current = self.params[VOUCHER]["strike"]
                tte_current = self.params[VOUCHER]["starting_time_to_expiry"]
                m_t_current = np.log(K_current / volcanic_mid_price) / np.sqrt(tte_current)
                v_t_current = BlackScholes.implied_volatility(mid_price, volcanic_mid_price, K_current, tte_current)
                
                voucher_orders = []
                
                residual = 0
                if "voucher_regression_data" in traderObject:
                    a, b, c = traderObject["voucher_regression_data"]["coeffs"]
                    v_t_predicted = a * (m_t_current ** 2) + b * m_t_current + c
                    residual = v_t_current - v_t_predicted
                
                    threshold = self.params[VOUCHER]['threshold']
                    if residual > threshold and rock_voucher_position > -self.LIMIT[VOUCHER]:
                        # sell voucher
                        if state.order_depths[VOUCHER].buy_orders:
                            best_bid = max(state.order_depths[VOUCHER].buy_orders.keys())
                            quantity = min(abs(rock_voucher_position - (-self.LIMIT[VOUCHER])), 10)
                            voucher_orders.append(Order(VOUCHER, best_bid, -quantity))
                    elif residual < -threshold and rock_voucher_position < self.LIMIT[VOUCHER]:
                        # buy voucher
                        if state.order_depths[VOUCHER].sell_orders:
                            best_ask = min(state.order_depths[VOUCHER].sell_orders.keys())
                            quantity = min(abs(self.LIMIT[VOUCHER] - rock_voucher_position), 10)
                            voucher_orders.append(Order(VOUCHER, best_ask, quantity))
                
                delta = BlackScholes.delta(mid_price, K_current, tte_current, v_t_current)
                if not (0 <= delta <= 1):  
                    delta = 0.5
                
                hedge_orders = []
                if rock_order_depth is not None:
                    hedge_orders = self.volcanic_rock_hedge_orders(
                        rock_order_depth,
                        state.order_depths[VOUCHER],
                        voucher_orders,  
                        rock_position,
                        rock_voucher_position,
                        delta
                    )

                if voucher_orders:
                    result[VOUCHER] = voucher_orders
                
                if hedge_orders:
                    if Product.VOLCANIC_ROCK in result:
                        result[Product.VOLCANIC_ROCK].extend(hedge_orders)
                    else:
                        result[Product.VOLCANIC_ROCK] = hedge_orders


        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
