from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import pandas as pd
import numpy as np
import jsonpickle




def features(product, state):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    bid_volume = np.sum([x[1]for x in buy_orders])                  
    bid_vwap = np.sum(list(map(lambda x: np.prod(x), buy_orders)))/bid_volume
    ask_volume = np.sum([x[1]for x in sell_orders])
    ask_vwap = np.sum(list(map(lambda x: np.prod(x), sell_orders)))/ask_volume
    vwap = (bid_vwap*np.sum([x[1]for x in buy_orders]) + ask_vwap * np.sum([x[1]for x in sell_orders])) \
    /(np.sum([x[1]for x in sell_orders])+np.sum([x[1]for x in buy_orders]))
    mid_price = (buy_orders[0][0]+sell_orders[0][0])/2
    volume_spread = (bid_volume - ask_volume)/(bid_volume+ask_volume)
    vwap_spread = vwap - mid_price
    return mid_price, buy_orders[0][0], sell_orders[0][0], vwap



def take_market_buy(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    sell_orders = order_depth.sell_orders
    sell_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(sell_orders.items()), key=lambda x:x[0])))
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    max_trade = limit[1]-position
    orders = []
    i = 0
    quantity = 0
    while i<len(sell_orders) and sell_orders[i][1]<max_trade:
        max_trade-=sell_orders[i][1]
        quantity+=sell_orders[i][1]
        i+=1
    if i == len(sell_orders):
        return [Order(product, sell_orders[-1][0], quantity)]
    else:
        return [Order(product, sell_orders[i][0], quantity+max_trade)]

def take_market_buy_level_1(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    sell_orders = order_depth.sell_orders
    sell_orders = list(map(lambda x:(x[0],abs(x[1])), sorted(list(sell_orders.items()), key=lambda x:x[0])))
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    max_trade = limit[1]-position
    quantity = min(max_trade, sell_orders[0][1])
    return [Order(product, sell_orders[-1][0], quantity)]
    
        

def take_market_sell(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    buy_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(buy_orders.items()), key=lambda x:-x[0])))
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    max_trade = position - limit[0]
    orders = []
    i = 0
    quantity = 0
    while i<len(buy_orders) and buy_orders[i][1]<max_trade:
        max_trade-=buy_orders[i][1]
        quantity+=buy_orders[i][1]
        i+=1
    if i == len(buy_orders):
        return [Order(product, buy_orders[-1][0], quantity)]
    else:
        return [Order(product, buy_orders[i][0], -(quantity+max_trade))]

def take_market_sell_level_1(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    buy_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(buy_orders.items()), key=lambda x:-x[0])))
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    max_trade = position - limit[0]
    i = 0
    quantity = min(max_trade,buy_orders[0][1])
    return [Order(product, buy_orders[-1][0], -quantity)]

def take_market_at_price(product, state, price, limit=(-50, 50)):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    orders = []
    if price>sell_orders[0][0]:
        max_trade = limit[1] - position
        if max_trade!=0:
            quantity = 0
            for i in range(len(sell_orders)):
                if sell_orders[i][0]<price:
                    best_price = sell_orders[i][0]
                    quantity += min(sell_orders[i][1], max_trade)
                    max_trade-=min(sell_orders[i][1], max_trade)
                    sell_orders[i][1]=sell_orders[i][1]-quantity
                    if max_trade==0:
                        break
                else:
                    break
            i = 0
            while sell_orders[i][1]==0:
                i+=1
            sell_orders = sell_orders[i:]
            assert sell_orders[0][1]!=0
            orders.append(Order(product, best_price, int(quantity)))
            print(f"Buy: ({best_price},{quantity})")
            position+=quantity
            update_state(state, product, position, buy_orders, sell_orders)
        return orders
    elif price<buy_orders[0][0]:
        max_trade = position-limit[0]
        if max_trade != 0:
            quantity = 0
            for i in range(len(buy_orders)):
                if buy_orders[i][0]>price:
                    best_price = buy_orders[i][0]
                    quantity += min(buy_orders[i][1], max_trade)
                    max_trade-=min(buy_orders[i][1], max_trade)
                    buy_orders[i][1]=buy_orders[i][1]-quantity
                    if max_trade == 0:
                        break
                else:
                    break
            i = 0
            while buy_orders[i][1]==0:
                i+=1
            buy_orders = buy_orders[i:]
            
            orders+=[Order(product, best_price, int(-quantity))]
            print(f"Sell: ({best_price},{-quantity})")
            position-=quantity
            update_state(state, product, position, buy_orders, sell_orders)
        return orders
    else:
        return orders    
    

def get_product_cost(state):
    ink_dict = {'position': 0, 'cost': 0}
    kelp_dict = {'position': 0, 'cost': 0}
    resin_dict = {'position': 0, 'cost': 0}
    if 'SQUID_INK' in state.own_trades:
        for trade in state.own_trades['SQUID_INK']:
            if trade.buyer:
                ink_dict['position']+=trade.quantity
                ink_dict['cost']+=trade.quantity*trade.price
            elif trade.seller:
                ink_dict['position']-=trade.quantity
                ink_dict['cost']-=trade.quantity*trade.price
    if 'KELP' in state.own_trades: 
        for trade in state.own_trades['KELP']:
            if trade.buyer:
                kelp_dict['position']+=trade.quantity
                kelp_dict['cost']+=trade.quantity*trade.price
            elif trade.seller:
                kelp_dict['position']-=trade.quantity
                kelp_dict['cost']-=trade.quantity*trade.price
    if 'RAINFOREST_RESIN' in state.own_trades:
        for trade in state.own_trades['RAINFOREST_RESIN']:
            if trade.buyer:
                resin_dict['position']+=trade.quantity
                resin_dict['cost']+=trade.quantity*trade.price
            elif trade.seller:
                resin_dict['position']-=trade.quantity
                resin_dict['cost']-=trade.quantity*trade.price
        
    return ink_dict, kelp_dict, resin_dict

def get_market_price(state):
    ink_dict = {'volume': 0, 'cost': 0}
    kelp_dict = {'volume': 0, 'cost': 0}
    resin_dict = {'volume': 0, 'cost': 0}
    
    if 'SQUID_INK' in state.market_trades:
        for trade in state.market_trades['SQUID_INK']:
            ink_dict['volume']+=trade.quantity
            ink_dict['cost']+=trade.quantity*trade.price
    if 'KELP' in state.market_trades:
        for trade in state.market_trades['KELP']:
            kelp_dict['volume']+=trade.quantity
            kelp_dict['cost']+=trade.quantity*trade.price
    if 'RAINFOREST_RESIN' in state.market_trades:
        for trade in state.market_trades['RAINFOREST_RESIN']:
            resin_dict['volume']+=trade.quantity
            resin_dict['cost']+=trade.quantity*trade.price
        
    return ink_dict['cost']/ink_dict['volume'] if ink_dict['volume'] else 0, \
            kelp_dict['cost']/kelp_dict['volume'] if kelp_dict['volume'] else 0, \
            resin_dict['cost']/resin_dict['volume'] if resin_dict['volume'] else 0




def make_market_bid(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    buy_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(buy_orders.items()), key=lambda x:-x[0])))
    
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    max_trade = limit[1]-position
    
    return [Order(product, buy_orders[0][0]+1, max_trade)]

def make_market_ask(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    sell_orders = order_depth.sell_orders
    sell_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(sell_orders.items()), key=lambda x:x[0])))
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    max_trade = position-limit[0]
    return [Order(product, sell_orders[0][0]-1, -max_trade)]


def make_market_bid_spread_1(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    buy_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(buy_orders.items()), key=lambda x:-x[0])))
    sell_orders = order_depth.sell_orders
    sell_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(sell_orders.items()), key=lambda x:x[0])))
    depth = sell_orders[0][0]-buy_orders[0][0] - 2
    
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    
    max_trade = limit[1]-position
    numbers = np.zeros(depth)
    i = 0
    while max_trade>0:
        numbers[i]+=1
        max_trade-=1
        i+=1
        i%=depth
    numbers = numbers.tolist()
    prices = (np.array(list(range(1,depth+1))[::-1])+buy_orders[0][0]).tolist()
    orders = []
    while numbers:
        number = numbers.pop()
        price = prices.pop()
        if number !=0:
            orders.append(Order(product, price, int(number)))
    return orders

def make_market_ask_spread_1(product, state, limit=(-50, 50)):
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    buy_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(buy_orders.items()), key=lambda x:-x[0])))
    sell_orders = order_depth.sell_orders
    sell_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(sell_orders.items()), key=lambda x:x[0])))
    depth = sell_orders[0][0]-buy_orders[0][0] - 2
    
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    
    max_trade = position-limit[0]
    numbers = np.zeros(depth)
    i = 0
    while max_trade>0:
        numbers[i]+=1
        max_trade-=1
        i+=1
        i%=depth
    numbers = numbers.tolist()
    prices = (-np.array(list(range(1,depth+1))[::-1])+sell_orders[0][0]).tolist()
    orders = []
    while numbers:
        number = numbers.pop()
        price = prices.pop()
        if number !=0:
            orders.append(Order(product, price, -int(number)))
    return orders

def make_market_bid_to_best(product, state, best, limit=(-50, 50)):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    depth = best - buy_orders[0][0]
    max_trade = limit[1]-position
    if depth==0:
        return [Order(product, buy_orders[0][0], max_trade)]
    numbers = np.zeros(depth)
    i = 0
    while max_trade>1:
        numbers[i]+=1
        max_trade-=1
        i+=1
        i%=depth
    numbers = numbers.tolist()
    prices = (np.array(list(range(1,depth+1)))+buy_orders[0][0]).tolist()
    orders = []
    while numbers:
        number = numbers.pop()
        price = prices.pop()
        if number !=0:
            orders.append(Order(product, price, int(number)))
            print(f"Bid: ({price},{number})")
    return orders

def make_market_ask_to_best(product, state, best, limit=(-50, 50)):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    depth = sell_orders[0][0]-best
    max_trade = position-limit[0]
    if depth==0:
        return [Order(product, sell_orders[0][0], -max_trade)]
    numbers = np.zeros(depth)
    i = 0
    while max_trade>0:
        numbers[i]+=1
        max_trade-=1
        i+=1
        i%=depth
    numbers = numbers.tolist()
    prices = (-np.array(list(range(1,depth+1)))+sell_orders[0][0]).tolist()
    orders = []
    while numbers:
        number = numbers.pop()
        price = prices.pop()
        if number !=0:
            orders.append(Order(product, price, int(-number)))
            print(f"Ask: ({price},{-number})")
    return orders

def update_state(state, product, position, buy_orders, sell_orders):
    state.position[product]=position
    state.order_depths[product].buy_orders = dict(buy_orders)
    state.order_depths[product].sell_orders = dict([[x[0],-x[1]] for x in sell_orders])

def get_order_lists(product, state):
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    sell_orders = order_depth.sell_orders
    buy_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(buy_orders.items()), key=lambda x:-x[0])))
    sell_orders = list(map(lambda x:[x[0],abs(x[1])], sorted(list(sell_orders.items()), key=lambda x:x[0])))
    if product in state.position:
        position = state.position[product]
    else:
        position = 0
    i=0
    while buy_orders[i][1]==0:
        i+=1
    buy_orders = buy_orders[i:]
    i=0
    while sell_orders[i][1]==0:
        i+=1
    sell_orders = sell_orders[i:]
    assert len(buy_orders)>0 and len(sell_orders)>0
    update_state(state, product, position, buy_orders, sell_orders)
    return position, buy_orders, sell_orders

def stable_price_take_make_market(product, state, limit=(-50, 50), price=10000):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    print(f"position: {position}, buy_orders: {buy_orders}, sell_orders:{sell_orders}")
    orders = []
    if buy_orders[0][0]>price:
        max_trade = position - limit[0]
        orders+=take_market_at_price(product, state, 10000, limit=(-50, 50))
        position, buy_orders, sell_orders = get_order_lists(product, state)
        max_trade = position - limit[0]
        if max_trade:
            orders+=make_market_ask_to_best(product, state, sell_orders[0][0]-1, limit=(-50, 50))
            #orders+=make_market_ask_to_best(product, state, 10001, limit=(-50, 50))
        return orders
            
    elif sell_orders[0][0]<price:
        max_trade = limit[1] - position
        orders+=take_market_at_price(product, state, 10000, limit=(-50, 50))
        position, buy_orders, sell_orders = get_order_lists(product, state)
        max_trade = position - limit[0]
        if max_trade:
            orders+=make_market_bid_to_best(product, state, buy_orders[0][0]+1, limit=(-50, 50))
            #orders+=make_market_bid_to_best(product, state, 9999, limit=(-50, 50))
        return orders
        
        
    else:
        if sell_orders[0][0]>10001:
            orders+=make_market_ask_to_best(product, state, sell_orders[0][0]-1, limit=(-50, 50))
            #orders+=make_market_ask_to_best(product, state, 10001, limit=(-50, 50))
        if buy_orders[0][0]<9999:
            orders+=make_market_bid_to_best(product, state, buy_orders[0][0]+1, limit=(-50, 50))
            #orders+=make_market_bid_to_best(product, state, 9999, limit=(-50, 50))
        return orders

def get_price_volume(product, state):
    pv=0
    v=0
    if product in state.market_trades:
        for trade in state.market_trades[product]:
            pv+=trade.price*trade.quantity
            v+= trade.quantity
        
    if product in state.own_trades:
        for trade in state.own_trades[product]:
            pv+=trade.price*trade.quantity
            v+= trade.quantity
    
    return [pv, v]




def convey_whole_trades(state, traderData):
    if state.traderData:
        market_trades, own_trades = jsonpickle.decode(state.traderData)
    else:
        market_trades = {}
        own_trades = {}
    for product in state.order_depths.keys():
        if product not in market_trades:
            market_trades[product] = []
        if product not in own_trades:
            own_trades[product] = []
        if product in state.market_trades:
            for trade in state.market_trades[product]:
                market_trades[product].append([trade.price, trade.quantity, trade.timestamp])
        if product in state.own_trades:    
            for trade in state.own_trades[product]:
                buyer = 0
                if trade.buyer!="":
                    buyer = 1
                elif trade.seller!="":
                    buyer = -1
                own_trades[product].append([trade.price, trade.quantity, trade.timestamp, buyer])
    traderData = jsonpickle.encode([market_trades, own_trades])
    return traderData

def get_momentum(pv):
    df = pd.DataFrame(np.array(pv), columns=['pv', 'v'])
    df['vwap'] = (df['pv']/df['v']).ffill()
    ma_s = df['vwap'].rolling(5).mean()
    ma_l = df['vwap'].rolling(10).mean()
    return (ma_s-ma_l).tolist()[-1]
    
def get_VM(pv, mid_price):
    df = pd.DataFrame(np.array(pv), columns=['pv', 'v'])
    return  df['pv'].iloc[-1] - mid_price*df['v'].iloc[-1]  

def get_OBI(product, state):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    bid_volume = sum([x[1] for x in buy_orders])
    ask_volume = sum([x[1] for x in sell_orders])
    return (bid_volume - ask_volume)/(bid_volume + ask_volume)

def get_vwap_spread(product, state):
    position, buy_orders, sell_orders = get_order_lists(product, state)
    bid_volume = sum([x[1] for x in buy_orders])
    ask_volume = sum([x[1] for x in sell_orders])
    pv_bid = sum([x[1]*x[0] for x in buy_orders])
    pv_ask = sum([x[1]*x[0] for x in sell_orders])
    vwap = (pv_bid+pv_ask)/(bid_volume+ask_volume)
    return vwap - (buy_orders[0][0]+sell_orders[0][0])/2

            
class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """    
        
        ink_dict, kelp_dict, resin_dict = get_product_cost(state)
        ink_price, kelp_price, resin_price = get_market_price(state)
        
        result = {'RAINFOREST_RESIN':[], 'SQUID_INK':[], 'KELP':[]}
        for product in state.order_depths.keys():
            if product == 'RAINFOREST_RESIN':
                result[product] = stable_price_take_make_market(product, state, limit=(-50, 50), price=10000)
            elif product=='SQUID_INK':
                """
                mid_price, best_bid, best_ask, vwap = ob_features(state.order_depths[product])
                my_cost = kelp_dict['cost']/kelp_dict['position'] if kelp_dict['position'] else 0
                
                if ink_price-mid_price>1 and best_ask-best_bid>1:
                    if my_cost<best_ask:
                        result[product] += make_market_ask(product, state, limit=(-50, 50))
                    if best_ask-best_bid>2:
                        result[product] += make_market_bid(product, state, limit=(-50, 50))
                elif ink_price-mid_price<-1 and best_ask-best_bid>1:
                    if my_cost>best_bid:
                        result[product] += make_market_bid(product, state, limit=(-50, 50))
                    if best_ask-best_bid>2:
                        result[product] += make_market_ask(product, state, limit=(-50, 50))
                """
            else:
                """
                mid_price, best_bid, best_ask, vwap = ob_features(state.order_depths[product])
                my_cost = kelp_dict['cost']/kelp_dict['position'] if kelp_dict['position'] else 0
                if vwap-mid_price>1:
                    result[product] = take_market_buy(product, state, limit=(-50, 50))
                elif vwap-mid_price<-1:
                    result[product] = take_market_sell(product, state, limit=(-50, 50))
                
                mid_price, best_bid, best_ask, vwap = ob_features(state.order_depths[product])
                my_cost = kelp_dict['cost']/kelp_dict['position'] if kelp_dict['position'] else 0
                if best_ask-best_bid>2:   
                    result[product] += make_market_ask(product, state, limit=(-50, 50))
                    result[product] += make_market_bid(product, state, limit=(-50, 50))
                
                """
                
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1 

                # Return the dict of orders
                # These possibly contain buy or sell orders
                # Depending on the logic above
        
        return result, conversions, traderData