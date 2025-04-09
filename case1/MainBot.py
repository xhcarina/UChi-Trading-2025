from typing import Optional
import asyncio
import argparse
import math
import numpy as np
import random
import os
from datetime import datetime
from utcxchangelib import xchange_client
from utcxchangelib.xchange_client import SWAP_MAP


SYMBOLS = ["APT", "DLR", "MKJ"]
ETFS = ["AKAV"]
ALL_SYMBOLS = SYMBOLS + ETFS

MAX_ORDER_SIZE = 40
MAX_OPEN_ORDERS = 50
MAX_ABSOLUTE_POSITION = 200
OUTSTANDING_VOLUME = 120

LOG_PATH = "./log"
os.makedirs(LOG_PATH, exist_ok=True)


class MyXchangeClient(xchange_client.XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.round = 0
        self.logfile = os.path.join(LOG_PATH, f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        #Fair price variables
        self.fair_price_APT = None    
        self.fair_price_DLR = None   
        self.fair_price_MKJ = None
        self.fair_price_AKAV = None
        self.fair_price_AKIM = None
        
        #News variables
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
        

        #Trading variables
        self.order_size = 10
        self.etf_margin = 5 # minimum amount needed to justify ETF arbitrage
        self.fade = 50
        self.min_margin = 1
        self.edge_sensitivity = 0.5
        self.slack = 4
        self.spreads = [5, 10, 15] 
        self.level_orders = 5  

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

    async def bot_place_order_safe(self, symbol, qty, side, price):
        if abs(self.positions[symbol]) + qty > MAX_ABSOLUTE_POSITION:
            print(f"Over max position for {symbol}")
            return
        if len(self.open_orders) >= MAX_OPEN_ORDERS:
            print(f"Over max open orders")
            return
        await self.place_order(symbol=symbol, qty=qty, side=side, px=price)
        self.log(f"Placed {side.name} {qty} {symbol} @ {price}")

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

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        """
        Called when a cancel order response is received.
        If success is True, it logs the cancellation and (if desired) removes the order from self.open_orders.
        If not successful, logs the error.
        """
        if order_id in self.open_orders:
            order = self.open_orders[order_id]  # Assume order structure: [order_request, pending_qty, is_market]
            if success:
                print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")
                self.log(f"Cancel response: Order {order_id} cancelled. Unfilled quantity: {order[1]}")
                # Remove the order from tracking because it was cancelled successfully.
                del self.open_orders[order_id]
            else:
                print(f"Cancel response: Order {order_id} cancellation failed - {error}")
                self.log(f"Cancel response: Order {order_id} cancellation failed with error: {error}")
        else:
            print(f"Cancel response received for unknown order {order_id}")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        """
        Called when an order is filled (partially or completely).
        Here, we simply log the fill event and print the updated positions.
        """
        # The base client has already updated self.positions.
        print("Order fill received for order", order_id)
        print("Filled quantity:", qty, "at price:", price)
        print("Updated positions:", self.positions)
        self.log(f"Order fill: Order {order_id} filled {qty} shares at {price}. New positions: {self.positions}")
        # Optionally, if you track partial fills within open_orders,
        # update or remove the order from self.open_orders accordingly.
        if order_id in self.open_orders:
            # could optionally remove the entry if fully filled.
            order = self.open_orders[order_id]
            if order[1] - qty <= 0:
                del self.open_orders[order_id]
            else:
                order[1] -= qty  # Reduce remaining quantity

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        """
        Called when an order is rejected.
        Logs the rejection reason and removes the order from our tracking.
        """
        print("Order rejected because of:", reason)
        self.log(f"Order {order_id} rejected: {reason}")
        # Remove the order from open_orders if it is tracked.
        if order_id in self.open_orders:
            del self.open_orders[order_id]


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # This function is called when an actual trade is executed.
        # It provides immediate market data such as the transaction price and volume.
        print(f"Trade executed for {symbol}: {qty} shares at {price}")
        # Here you could update your fair price estimation based on recent trade activity.
        # For example:
        #fair_price_based_on_trade = <YOUR_FORMULA_HERE>
        # You might combine recent trade prices and volumes to adjust your short-term fair value.
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
            self.fair_price_MKJ = None 
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

        self.fair_price_DLR = 100 * probability
        print(f"[timestamp: {timestamp}] Updated DLR fair price: {self.fair_price_DLR:.2f} "
            f"(Probability: {probability:.2%}, remaining events: {remaining_events})")




    async def trade(self):
        while True:
            # TODO: replace mid-price with actual fair_price
            fair_price = {"APT": self.fair_price_APT, "DLR": self.fair_price_DLR, "MKJ": self.fair_price_MKJ}

            # ETF Arbitrage
            await self.bot_place_arbitrage_order()

            # Market Making with Fade and Edge
            for symbol in SYMBOLS:
                position = self.positions[symbol]
                fade_adj = -self.fade * (1 if position > 0 else -1) * math.log2(1 + abs(position) / MAX_ABSOLUTE_POSITION)

                for side in [xchange_client.Side.BUY, xchange_client.Side.SELL]:
                    activity = self.get_market_activity_level(symbol, side)
                    edge = max(int(round(self.min_margin + self.slack / 2 * (math.tanh(-4 * self.edge_sensitivity * activity + 2) + 1 ))), 1)

                    base_price = fair_price[symbol] + fade_adj
                    price = round(base_price - edge) if side == xchange_client.Side.BUY else round(base_price + edge)
                    await self.bot_place_order_safe(symbol, self.order_size, side, price)

                # Level Orders
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


