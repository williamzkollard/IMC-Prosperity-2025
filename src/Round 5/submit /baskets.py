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


        

class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"


PARAMS = {
    Product.SPREAD1: {
        "default_spread_mean": 50,
        "default_spread_std": 80,
        "spread_window": 55,
        "diff_threshold": 48,
        "target_position": 90,
        "exit_threshold": 0
    },
    Product.SPREAD2: {
        "default_spread_mean": 42,
        "default_spread_std": 50,
        "spread_window": 60,
        "diff_threshold": 42,
        "target_position": 90,
        "exit_threshold": 0
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



class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            }



    ########################################################################
    #                     METHODS FOR BASKETS
    ########################################################################

    def get_microprice(self, order_depth):
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])[:3]  # Top 3 bids (highest first)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])[:3]  # Top 3 asks (lowest first)


        if not buy_orders or not sell_orders:
            return None  # Only bids or only asks present
    
        total_price_volume = 0
        total_volume = 0

        for price, volume in buy_orders:
            abs_vol = abs(volume)
            total_price_volume += price * abs_vol
            total_volume += abs_vol

        for price, volume in sell_orders:
            abs_vol = abs(volume)
            total_price_volume += price * abs_vol
            total_volume += abs_vol

        if total_volume == 0:
            return None  # No valid price info

        return total_price_volume / total_volume



    
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
        

        
        diff_threshold = ( spread - self.params[SPREAD]["default_spread_mean"])
        
        if diff_threshold >= self.params[SPREAD]["diff_threshold"]:
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
        
        if diff_threshold <= -self.params[SPREAD]["diff_threshold"]:
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

        # Exit logic
        exit_threshold = self.params[SPREAD].get("exit_threshold", 0)
        entry = spread_data.get("entry", None)

        if abs(diff_threshold) < exit_threshold and picnic_position != 0 and entry:
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

        spread_data["prev_zscore"] = diff_threshold
        return None
        
   


    def update_price_history(self, product, mid_price, traderObject):
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



        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
