from typing import Optional
import asyncio
import argparse
import math
import numpy as np
import random
import os
from collections import defaultdict
from collections import deque
from SlidingWindow import find_fair_price
from datetime import datetime
from utcxchangelib import xchange_client
from utcxchangelib.xchange_client import SWAP_MAP, Side
from params_gui import ParametersGUI


SYMBOLS = ["APT", "DLR", "MKJ"]
ETFS = ["AKAV"]
ALL_SYMBOLS = SYMBOLS + ETFS

MAX_ORDER_SIZE = 40
MAX_OPEN_ORDERS = 50
MAX_ABSOLUTE_POSITION = 200
OUTSTANDING_VOLUME = 120
SLIDINGWINDOWCAP = 13

LOG_PATH = "./log"
os.makedirs(LOG_PATH, exist_ok=True)


class OrderBookAnalyzer:
    """Handles order book analysis and fair price calculation for a symbol"""
    
    def __init__(self, symbol: str, max_window_size: int = 20):
        self.symbol = symbol
        self.max_window_size = max_window_size
        self.sliding_window = deque(maxlen=max_window_size)
        self.last_fair_price = None
        self.last_cleaned_orders = None
        
    def add_order(self, price: float, volume: int, side: str):
        """Add a valid order to the sliding window"""
        if price is not None and volume > 0:
            self.sliding_window.append((price, volume, side))
            
    def clean_orders(self, orders: list) -> list:
        """Validate and clean order data"""
        cleaned = []
        for order in orders:
            if (isinstance(order, tuple) and len(order) == 3 and
                isinstance(order[0], (int, float)) and
                isinstance(order[1], (int, float)) and
                isinstance(order[2], str)):
                cleaned.append(order)
        return cleaned
        
    def calculate_fair_price(self, orders: list) -> float:
        """Calculate fair price using parentheses matching algorithm"""
        if not orders:
            return None
            
        # Sort orders by price
        sorted_orders = sorted(orders, key=lambda x: x[0])
        
        # Create parentheses string
        s = ""
        for price, volume, side in sorted_orders:
            if side == "bid":
                s += "(" * int(volume)
            else:  # ask
                s += ")" * int(volume)
                
        if not s:
            return None
            
        # Find optimal price point
        min_diff = float('inf')
        fair_price_idx = 0
        
        for i in range(len(s)):
            O = s[:i].count("(")
            C = s[i:].count(")")
            diff = abs(O - C)
            
            if diff < min_diff:
                min_diff = diff
                fair_price_idx = i
                
        # Map index back to price
        current_idx = 0
        for price, volume, side in sorted_orders:
            current_idx += int(volume)
            if current_idx > fair_price_idx:
                return price
                
        return sorted_orders[-1][0]
        
    def update_fair_price(self) -> float:
        """Update and return the current fair price"""
        current_orders = list(self.sliding_window)
        
        if not current_orders:
            return self.last_fair_price
            
        if current_orders != self.last_cleaned_orders:
            cleaned_orders = self.clean_orders(current_orders)
            if cleaned_orders:
                new_price = self.calculate_fair_price(cleaned_orders)
                if new_price is not None:
                    self.last_fair_price = new_price
            self.last_cleaned_orders = current_orders
            
        return self.last_fair_price

class MyXchangeClient(xchange_client.XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.round = 0
        self.logfile = os.path.join(LOG_PATH, f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        # Fair price variables
        self.fair_price_APT = None    
        self.fair_price_DLR = None   
        self.fair_price_MKJ = None
        self.fair_price_AKAV = None
        self.fair_price_AKIM = None

        # ETF arbitrage variables
        self.prev_day_close_AKAV = None
        self.curr_day_open_AKIM = None
        self.etf_margin = 5 # minimum amount needed to justify ETF arbitrage
        self.inverse_eft_margin = 5 # minimum amount needed to justify inverse ETF arbitrage

        # Sliding window for order book analysis
        self.sliding_window = deque(maxlen=SLIDINGWINDOWCAP)

        # Container for open inverse arb positions
        self.inverse_arb_positions = []
        self.orderid_to_positionid = {}
        self.order_fills_qty = defaultdict(int)      # {order_id: total shares filled so far}
        self.order_fills_value = defaultdict(float)  # {order_id: total fill-value so far (qty * price)}

        # News variables
        self.apt_earnings = None
        self.dlr_cumulative = 0 
        self.dlr_event_count = 0
        self.total_events_per_round = 10 * 5  # 10 days * 5 events per day = 50 events
        self.alpha = 1.0630449594499
        self.sigma = 0.006
        self.ln_mu = math.log(self.alpha)  # constant, fixed value
        self.ln_sigma = self.sigma           # constant, fixed value
        self.pe_ratio_APT = 10 #Given on Ed
        self.mkj_news_events = []     # List of unstructured news events for MKJ

        # Trading variables
        # Trading variables
        self.order_size = 10  # Default size for market making orders (recommended: 5-20)
        # Lower bound (5): More conservative, less risk but lower profit potential
        # Upper bound (20): More aggressive, higher risk but better profit potential
        # Current value (10): Balanced approach between risk and reward

        self.fade = 50  # Price adjustment factor based on position size to prevent overexposure (recommended: 30-70)
        # Lower bound (30): Less position-based price adjustment, more aggressive position building
        # Upper bound (70): More conservative position management, quicker to reduce exposure
        # Current value (50): Moderate position management

        self.min_margin = 1  # Minimum profit margin required for orders (recommended: 0.5-2)
        # Lower bound (0.5): More aggressive, willing to take smaller profits
        # Upper bound (2): More selective, only takes trades with higher profit potential
        # Current value (1): Balanced profit target

        self.edge_sensitivity = 0.5  # Controls how aggressively we adjust prices based on market activity (recommended: 0.2-1.0)
        # Lower bound (0.2): Less responsive to market activity, more stable pricing
        # Upper bound (1.0): Very responsive to market activity, more dynamic pricing
        # Current value (0.5): Moderate market responsiveness

        self.slack = 4  # Additional buffer added to minimum margin for order placement (recommended: 2-6)
        # Lower bound (2): Tighter spreads, more competitive but less profit per trade
        # Upper bound (6): Wider spreads, less competitive but higher profit per trade
        # Current value (4): Balanced spread width

        self.spreads = [5, 10, 15]  # Different spread levels for tiered order placement (recommended: keep spreads within 2-20)
        # Lower bound (2): Very tight spreads, high competition, lower profit per trade
        # Upper bound (20): Very wide spreads, less competition, higher profit per trade
        # Current values [5,10,15]: Progressive spread levels for different market conditions

        self.level_orders = 5  # Number of orders to place at each spread level (recommended: 3-8)
        # Lower bound (3): Less market presence, lower risk but less profit potential
        # Upper bound (8): Strong market presence, higher risk but better profit potential
        # Current value (5): Moderate market presence

        # Initialize order book analyzers
        self.mkj_analyzer = OrderBookAnalyzer("MKJ", SLIDINGWINDOWCAP)

    def log(self, msg):
        with open(self.logfile, "a") as f:
            f.write(f"[Round {self.round}] {msg}\n")

    def get_market_activity_level(self, symbol, side):
        book = self.order_books[symbol]
        orders = book.bids if side == xchange_client.Side.BUY else book.asks
        midpoint = self.get_mid_price(symbol)
        edge_window = self.min_margin + self.slack
        return sum(1 for px in orders if abs(px - midpoint) <= edge_window)

    def get_mid_price(self, symbol):
        bids = self.order_books[symbol].bids
        asks = self.order_books[symbol].asks
        if not bids or not asks:
            return 0
        return (max(bids.keys()) + min(asks.keys())) / 2
    
    def get_best_bid(self, symbol: str) -> float:
        """Get best bid price with safety checks"""
        bids = self.order_books[symbol].bids
        return max(bids.keys()) if bids else 0.0

    def get_best_ask(self, symbol: str) -> float:
        """Get best ask price with safety checks"""
        asks = self.order_books[symbol].asks
        return min(asks.keys()) if asks else 0.0

    async def bot_place_order_safe(self, symbol, qty, side, price):
        if abs(self.positions[symbol]) + qty > MAX_ABSOLUTE_POSITION:
            #print(f"Over max position for {symbol}")
            return None
        if len(self.open_orders) >= MAX_OPEN_ORDERS:
            #print(f"Over max open orders")
            return None
        order_id = await self.place_order(symbol=symbol, qty=qty, side=side, px=price)
        #self.log(f"Placed {side.name} {qty} {symbol} @ {price}")
        return order_id

    async def bot_place_arbitrage_order(self):
        '''
            profit off difference between nav and etf
        '''
        # TODO: consider order of buying/selling during arbitrage
        # TODO: handle risk limit block

        # Get best bid/ask prices for components
        apt_bid = max(self.order_books["APT"].bids.keys()) if self.order_books["APT"].bids else 0
        apt_ask = min(self.order_books["APT"].asks.keys()) if self.order_books["APT"].asks else 0
        dlr_bid = max(self.order_books["DLR"].bids.keys()) if self.order_books["DLR"].bids else 0
        dlr_ask = min(self.order_books["DLR"].asks.keys()) if self.order_books["DLR"].asks else 0
        mkj_bid = max(self.order_books["MKJ"].bids.keys()) if self.order_books["MKJ"].bids else 0
        mkj_ask = min(self.order_books["MKJ"].asks.keys()) if self.order_books["MKJ"].asks else 0

        # NAV to create AKAV (buy components at ask prices)
        nav_create = apt_ask + dlr_ask + mkj_ask
        # NAV to redeem AKAV (sell components at bid prices)
        nav_redeem = apt_bid + dlr_bid + mkj_bid

        akav_bid = max(self.order_books["AKAV"].bids.keys()) if self.order_books["AKAV"].bids else 0
        akav_ask = min(self.order_books["AKAV"].asks.keys()) if self.order_books["AKAV"].asks else 0

        qty = 1 # TODO: change qty dynamically?

        if akav_bid - nav_create - SWAP_MAP["toAKAV"].cost > self.etf_margin:
            # ETF overpriced -> sell AKAV, buy components, swap from stocks to ETF
            # Buy components at best ask (or better)
            # TODO: give slightly better bid/ask to ensure order gets filled?
            await self.bot_place_order_safe("APT", qty, xchange_client.Side.BUY, apt_ask) 
            await self.bot_place_order_safe("DLR", qty, xchange_client.Side.BUY, dlr_ask)
            await self.bot_place_order_safe("MKJ", qty, xchange_client.Side.BUY, mkj_ask)
            # Sell AKAV at best bid (or better)
            await self.bot_place_order_safe("AKAV", qty, xchange_client.Side.SELL, akav_bid)
            await self.place_swap_order("toAKAV", qty)  # Convert components to AKAV
        elif akav_ask - nav_redeem - SWAP_MAP["fromAKAV"].cost > self.etf_margin:
            # ETF underpriced -> buy AKAV, sell components, swap from ETF
                # Buy AKAV at best ask (or better)
            await self.bot_place_order_safe("AKAV", qty, xchange_client.Side.BUY, akav_ask)
            # Sell components at best bid (or better)
            await self.bot_place_order_safe("APT", qty, xchange_client.Side.SELL, apt_bid)
            await self.bot_place_order_safe("DLR", qty, xchange_client.Side.SELL, dlr_bid)
            await self.bot_place_order_safe("MKJ", qty, xchange_client.Side.SELL, mkj_bid)
            await self.place_swap_order("fromAKAV", qty)  # Redeem AKAV to components

        # akav_nav = sum(fair_price[s] for s in SYMBOLS)
        # price = fair_price["AKAV"]
        # diff = price - akav_nav
        # swap_fee = SWAP_MAP["toAKAV"].cost if diff > 0 else SWAP_MAP["fromAKAV"].cost

        # if abs(diff) > self.etf_margin + swap_fee:
        #     qty = 1 # TODO: change qty dynamically?
        #     if diff > 0:
        #         # ETF overpriced -> sell AKAV, buy components, swap from stocks to ETF
        #         await self.bot_place_order_safe("AKAV", qty, xchange_client.Side.SELL, round(price))
        #         for s in SYMBOLS:
        #             await self.bot_place_order_safe(s, qty, xchange_client.Side.BUY, round(fair_price[s]))
        #         await self.place_swap_order("toAKAV", qty)
        #     else:
        #         # ETF underpriced -> buy AKAV, sell components, swap from ETF
        #         await self.bot_place_order_safe("AKAV", qty, xchange_client.Side.BUY, round(price))
        #         for s in SYMBOLS:
        #             await self.bot_place_order_safe(s, qty, xchange_client.Side.SELL, round(fair_price[s]))
        #         await self.place_swap_order("fromAKAV", qty)
        #     self.log(f"Arbitrage opportunity: ETF {'over' if diff > 0 else 'under'}priced by {diff:.2f}")
    
    async def update_akim_fair_value(self):
        """Calculate AKIM's fair value based on AKAV's intraday movement"""
        try:
            if self.prev_day_close_AKAV is None or self.curr_day_open_AKIM is None:
                return

            current_akav = self.get_mid_price("AKAV")
            if current_akav is None or current_akav == 0 or self.prev_day_close_AKAV == 0:
                return
                
            akav_pct_change = (current_akav - self.prev_day_close_AKAV) / self.prev_day_close_AKAV
            self.fair_price_AKIM = self.curr_day_open_AKIM * (1 - akav_pct_change)
            
            print(f"[update_akim_fair_value] Updated AKIM fair value to: {self.fair_price_AKIM}", flush=True)
        except Exception as e:
            print(f"[update_akim_fair_value] Error updating AKIM fair value: {e}", flush=True)

    def calculate_hedge_ratio(self) -> float:
        """Calculate dynamic hedge ratio based on current market prices"""
        akav_price = self.get_mid_price("AKAV")
        akim_price = self.get_mid_price("AKIM")
        
        if akim_price == 0:
            return 0.0
            
        return akav_price / akim_price

    async def bot_inverse_arbitrage_order(self):
        try:
            akim_bid = self.get_best_bid("AKIM")
            akim_ask = self.get_best_ask("AKIM")
            akav_bid = self.get_best_bid("AKAV")
            akav_ask = self.get_best_ask("AKAV")

            await self.update_akim_fair_value()

            # Skip if we don't have valid prices yet
            if self.fair_price_AKIM is None:
                return

            hedge_ratio = self.calculate_hedge_ratio()
            
            akav_qty = 10
            akim_qty = round(akav_qty * hedge_ratio)

            if akim_bid - self.fair_price_AKIM > self.inverse_eft_margin:
                # AKIM is overpriced --> sell AKIM and sell AKAV to hedge
                order_id_akim = await self.bot_place_order_safe("AKIM", akim_qty, xchange_client.Side.SELL, akim_bid)
                order_id_akav = await self.bot_place_order_safe("AKAV", akav_qty, xchange_client.Side.SELL, akav_bid)

                if order_id_akim and order_id_akav:
                    # We opened a new position; store it
                    pos_id = len(self.inverse_arb_positions)
                    self.inverse_arb_positions.append({
                        "pos_id": pos_id,
                        "is_open": True,
                        "akim_side": xchange_client.Side.SELL,
                        "akim_qty": akim_qty,
                        "akim_avg_px": 0.0,  # we'll update after fills
                        "akav_side": xchange_client.Side.SELL,
                        "akav_qty": akav_qty,
                        "akav_avg_px": 0.0,
                        "target_profit": akim_bid - self.fair_price_AKIM
                    })
                    # Map the new orders to this position
                    self.orderid_to_positionid[order_id_akim] = pos_id
                    self.orderid_to_positionid[order_id_akav] = pos_id
            
            if self.fair_price_AKIM - akim_ask > self.inverse_eft_margin:
                # AKIM is underpriced --> buy AKIM and buy AKAV to hedge
                order_id_akim = await self.bot_place_order_safe("AKIM", akim_qty, xchange_client.Side.BUY, akim_ask)
                order_id_akav = await self.bot_place_order_safe("AKAV", akav_qty, xchange_client.Side.BUY, akav_ask)

                if order_id_akim and order_id_akav:
                    pos_id = len(self.inverse_arb_positions)
                    self.inverse_arb_positions.append({
                        "pos_id": pos_id,
                        "is_open": True,
                        "akim_side": xchange_client.Side.BUY,
                        "akim_qty": akim_qty,
                        "akim_avg_px": 0.0,
                        "akav_side": xchange_client.Side.BUY,
                        "akav_qty": akav_qty,
                        "akav_avg_px": 0.0,
                        "target_profit": self.fair_price_AKIM - akim_ask
                    })
                    # Map the orders
                    self.orderid_to_positionid[order_id_akim] = pos_id
                    self.orderid_to_positionid[order_id_akav] = pos_id
        except Exception as e:
            print(f"[bot_inverse_arbitrage_order] Error: {e}", flush=True)

    async def check_and_close_inverse_positions(self):
        """
        Continuously monitor open inverse arb positions to see if
        we've captured enough spread to exit, or if we want to close at EOD.
        """
        akim_best_bid = self.get_best_bid("AKIM")
        akim_best_ask = self.get_best_ask("AKIM")
        akav_best_bid = self.get_best_bid("AKAV")
        akav_best_ask = self.get_best_ask("AKAV")

        for pos in self.inverse_arb_positions:
            if not pos["is_open"]:
                continue

            # Calculate current liquidation PnL
            # If side is SELL => we shorted, so we buy back at ask to close the short
            # If side is BUY  => we are long, so we sell at bid to close the long

            if pos["akim_side"] == xchange_client.Side.SELL:
                # We sold at pos["akim_avg_px"], to close short we buy at best_ask
                akim_pnl = (pos["akim_avg_px"] - akim_best_ask) * pos["akim_qty"]
            else:
                # We bought at pos["akim_avg_px"], to close the long we sell at best_bid
                akim_pnl = (akim_best_bid - pos["akim_avg_px"]) * pos["akim_qty"]

            if pos["akav_side"] == xchange_client.Side.SELL:
                akav_pnl = (pos["akav_avg_px"] - akav_best_ask) * pos["akav_qty"]
            else:
                akav_pnl = (akav_best_bid - pos["akav_avg_px"]) * pos["akav_qty"]

            total_pnl = akim_pnl + akav_pnl

            # If we've hit target profit (or if near the close, etc.), close
            if total_pnl >= pos["target_profit"]:
                self.log(f"Closing Inverse Arb position {pos['pos_id']} at PnL: {total_pnl:.2f}")
                
                # Place orders to flatten
                if pos["akim_side"] == xchange_client.Side.SELL:
                    # Close short by buying
                    await self.bot_place_order_safe("AKIM", pos["akim_qty"], xchange_client.Side.BUY, akim_best_ask)
                else:
                    # Close long by selling
                    await self.bot_place_order_safe("AKIM", pos["akim_qty"], xchange_client.Side.SELL, akim_best_bid)

                if pos["akav_side"] == xchange_client.Side.SELL:
                    await self.bot_place_order_safe("AKAV", pos["akav_qty"], xchange_client.Side.BUY, akav_best_ask)
                else:
                    await self.bot_place_order_safe("AKAV", pos["akav_qty"], xchange_client.Side.SELL, akav_best_bid)

                pos["is_open"] = False


    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("Order fill received. Updated positions:", self.positions)
        pass
    
    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("Order rejected because of:", reason)
        pass

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # This function is called when an actual trade is executed.
        # It provides immediate market data such as the transaction price and volume.
        #print(f"Trade executed for {symbol}: {qty} shares at {price}")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass




    async def bot_handle_news(self, news_release: dict):
        """        
        For APT (earnings news): earnings news is received twice per day.
        For DLR (petition news): petition news is received five times per day.
        """
        timestamp = news_release["timestamp"]
        news_type = news_release["kind"]
        news_data = news_release["new_data"]

        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            asset = news_data["asset"]
            if subtype == "earnings" and asset == "APT":
                earnings = news_data["value"]
                self.apt_earnings = earnings
                self.fair_price_APT = earnings * self.pe_ratio_APT #earnings * constant_PE_ratio
                print(f"[{timestamp}] APT Earnings Update: {earnings}")
            elif subtype == "petition" and asset == "DLR":
                # Process DLR signature update news.
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]
                print(f"[{timestamp}] DLR: new signatures: {new_signatures}, cumulative: {cumulative}")
                self.dlr_cumulative = cumulative
                # Increment event counter for DLR news events.
                self.dlr_event_count += 1
                # Update our fair price estimate using the fixed lognormal parameters.
                self.update_dlr_fair_price(cumulative, timestamp)
            else:
                print(f"[{timestamp}] Received structured news for asset {asset} with subtype {subtype}")
        else:
            content = news_data.get("content", "")
            self.mkj_news_events.append((timestamp, content))
            # Don't reset fair price here, let sliding window handle it
            # self.fair_price_MKJ = None 

            # Get market values for AKAV and AKIM
            if "EOD - AKIM has rebalanced" in content:
                self.prev_day_close_AKAV = self.get_mid_price("AKAV")
                self.curr_day_open_AKIM = self.get_mid_price("AKIM")
            #print(f"[{timestamp}] MKJ Unstructured News: {content}")
            #Carina still working on more advanded stuffs (only updated the certain parts to the git)



    def monte_carlo_probability(self, current: int, steps: int, target: int, alpha: float, sigma: float, n_sims: int = 10000) -> float:
        """
        Estimate the probability of reaching the target number of signatures using Monte Carlo simulation.
        
        Args:
            current: current cumulative signatures
            steps: how many signature events remain
            target: the goal (100000)
            alpha: median multiplicative growth per step
            sigma: volatility of the lognormal growth
            n_sims: number of Monte Carlo trials

        Returns:
            Estimated probability of reaching the target by the end of steps
        """
        if current >= target:
            return 1.0
        if steps == 0:
            return 0.0

        # lognormal distribution: np.random.lognormal takes in mean and sigma of the *log*
        log_mu = np.log(alpha)

        # Simulate n_sims paths, each with 'steps' multiplicative jumps
        random_factors = np.random.lognormal(mean=log_mu, sigma=sigma, size=(n_sims, steps))
        cumulative_ends = current * np.prod(random_factors, axis=1)

        # Compute how many trials reached the threshold
        successes = np.sum(cumulative_ends >= target)
        return successes / n_sims


    def update_dlr_fair_price(self, cumulative: int, timestamp: int):
        """
        Update the fair price of DLR by calling a Monte Carlo simulation to estimate the probability of reaching
        100,000 signatures.
        """
        threshold = 100000
        required = max(threshold - cumulative, 0)
        remaining_events = self.total_events_per_round - self.dlr_event_count

        if required == 0:
            probability = 1.0
        elif remaining_events <= 0:
            probability = 0.0
        else:
            # Call the Monte Carlo simulation to compute the probability.
            probability = self.monte_carlo_probability(cumulative, remaining_events, threshold, self.alpha, self.sigma, n_sims=10000)

        self.fair_price_DLR = 10000 * probability
        print(f"[timestamp: {timestamp}] Updated DLR fair price: {self.fair_price_DLR:.2f} "
            f"(Probability: {probability:.2%}, remaining events: {remaining_events})")




    async def trade(self):
        while True:
            # TODO: replace mid-price with actual fair_price
            fair_price = {"APT": self.fair_price_APT, "DLR": self.fair_price_DLR, "MKJ": self.fair_price_MKJ}

            # Start MKJ fair price calculation
            asyncio.create_task(self.MKJ_fair_price_SW())

            # ETF Arbitrage
            await self.bot_place_arbitrage_order()

            # Inverse ETF Arbitrage
            await self.bot_inverse_arbitrage_order()
            await self.check_and_close_inverse_positions()

            # Market Making with Fade and Edge
            for symbol in SYMBOLS:
                # Skip if we don't have a fair price yet
                if fair_price[symbol] is None:
                    continue
                    
                position = self.positions[symbol]
                fade_adj = -self.fade * (1 if position > 0 else -1) * math.log2(1 + abs(position) / MAX_ABSOLUTE_POSITION)
                base_price = fair_price[symbol] + fade_adj

                for side in [xchange_client.Side.BUY, xchange_client.Side.SELL]:
                    activity = self.get_market_activity_level(symbol, side)
                    edge = max(int(round(self.min_margin + self.slack / 2 * (math.tanh(-4 * self.edge_sensitivity * activity + 2) + 1))), 1)
                    
                    # If near position limit, adjust edge to make the opposing orders more attractive
                    if abs(self.positions[symbol]) >= 180:
                        edge += 3
                    
                    price = round(base_price - edge) if side == xchange_client.Side.BUY else round(base_price + edge)
                    await self.bot_place_order_safe(symbol, self.order_size, side, price)

                # Level Orders (similar adjustments can be applied here if desired)
                for level in range(len(self.spreads)):
                    spread = self.spreads[level]
                    bid = round(fair_price[symbol] + fade_adj - edge - spread)
                    ask = round(fair_price[symbol] + fade_adj + edge + spread)
                    await self.bot_place_order_safe(symbol, 2, xchange_client.Side.BUY, bid)
                    await self.bot_place_order_safe(symbol, 2, xchange_client.Side.SELL, ask)

            self.round += 1
            await asyncio.sleep(1)

    # async def view_books(self):
    #     while True:
    #         await asyncio.sleep(3)
    #         for security, book in self.order_books.items():
    #             sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
    #             sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
    #             print(f"Bids for {security}:\n{sorted_bids}")
    #             print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self, user_interface):
        asyncio.create_task(self.trade())

        # This is where Phoenixhood will be launched if desired. There is no need to change these
        # lines, you can either remove the if or delete the whole thing depending on your purposes.
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())

        await self.connect()

    async def extract_all_from_book(self, symbol: str) -> list:
        """
        Extract all orders (price, volume, order_type) from the order book for the given symbol.
        Records both bids and asks.
        """
        try:
            book = self.order_books[symbol]
            orders = []
            for price, volume in book.bids.items():
                if volume != 0:
                    orders.append((price, volume, "bid"))
            for price, volume in book.asks.items():
                if volume != 0:
                    orders.append((price, volume, "ask"))
            return orders
        except Exception as e:
            print(f"[extract_all_from_book] Error extracting orders for {symbol}: {e}")
            return []

    async def collect_top_order_updates(self, symbol: str):
        """Collect top-of-book updates for a symbol"""
        analyzer = self.mkj_analyzer if symbol == "MKJ" else None
        if not analyzer:
            return
            
        while True:
            try:
                book = self.order_books[symbol]
                
                # Get top bid and ask
                top_bid_price = max(book.bids) if book.bids else None
                top_ask_price = min(book.asks) if book.asks else None
                
                # Add valid orders to analyzer
                if top_bid_price is not None:
                    top_bid_volume = book.bids[top_bid_price]
                    analyzer.add_order(top_bid_price, top_bid_volume, "bid")
                    
                if top_ask_price is not None:
                    top_ask_volume = book.asks[top_ask_price]
                    analyzer.add_order(top_ask_price, top_ask_volume, "ask")
                    
            except Exception as e:
                print(f"[collect_top_order_updates] Error for {symbol}: {e}")
                
            await asyncio.sleep(0.1)
            
    async def MKJ_fair_price_SW(self):
        """Update MKJ fair price using sliding window"""
        # Start order collection
        asyncio.create_task(self.collect_top_order_updates("MKJ"))
        
        while True:
            try:
                # Update fair price
                self.fair_price_MKJ = self.mkj_analyzer.update_fair_price()
                if self.fair_price_MKJ is not None:
                    print(f"[MKJ_fair_price_SW] Updated fair price: {self.fair_price_MKJ}", flush=True)
                    
            except Exception as e:
                print(f"[MKJ_fair_price_SW] Error: {e}", flush=True)
                
            await asyncio.sleep(0.5)


async def main(user_interface: bool):
    # SERVER = '127.0.0.1:8000'   # run locally
    SERVER = '3.138.154.148:3333'
    my_client = MyXchangeClient(SERVER,"chicago10","Gaw%3opFxg")
    await my_client.start(user_interface)
    return

if __name__ == "__main__":

    # This parsing is unnecessary if you know whether you are using Phoenixhood.
    # It is included here so you can see how one might start the API.

    parser = argparse.ArgumentParser(
        description="Script that connects client to exchange, runs algorithmic trading logic, and optionally deploys Phoenixhood"
    )

    parser.add_argument("--phoenixhood", required=False, default=False, type=bool, help="Starts phoenixhood API if true")
    args = parser.parse_args()

    user_interface = args.phoenixhood

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main(user_interface))


