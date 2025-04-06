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
        
        #News variables
        self.apt_earnings = None

        self.dlr_cumulative = 0 
        self.dlr_event_count = 0
        self.total_events_per_round = 10 * 5  # 10 days * 5 events per day = 50 events
   
   
        self.log_increments = []  # store log(new_signatures)
        self.ln_mu = 10.0    # initial guess; adjust with calibration data
        self.ln_sigma = 0.5  # initial guess; adjust with calibration data

        self.mkj_news_events = []     # List of unstructured news events for MKJ
        

        #Trading variables
        self.order_size = 10
        self.etf_margin = 5 # minimum amount needed to justify ETF arbitrage
        self.fade = 2
        self.min_margin = 1
        self.edge_sensitivity = 0.5
        self.slack = 4
        self.spreads = [5, 10] 
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
            print(f"Over max open orders for {symbol}")
            return
        await self.place_order(symbol=symbol, qty=qty, side=side, px=price)
        self.log(f"Placed {side.name} {qty} {symbol} @ {price}")

    async def bot_place_arbitrage_order(self, fair_price):
        # TODO: account for different swap fees (to and from AKAV)
        swap_fee = SWAP_MAP["toAKAV"].cost
        akav_nav = sum(fair_price[s] for s in SYMBOLS)
        price = fair_price["AKAV"]
        diff = price - akav_nav

        if abs(diff) > self.etf_margin + swap_fee:
            qty = 1 # TODO: change qty dynamically?
            if diff > 0:
                # ETF overpriced -> sell AKAV, buy components, swap from ETF to stocks
                await self.bot_place_order_safe("AKAV", qty, xchange_client.Side.SELL, round(price))
                for s in SYMBOLS:
                    await self.bot_place_order_safe(s, qty, xchange_client.Side.BUY, round(fair_price[s]))
                await self.place_swap_order("fromAKAV", qty)
            else:
                # ETF underpriced -> buy AKAV, sell components, swap to ETF
                await self.bot_place_order_safe("AKAV", qty, xchange_client.Side.BUY, round(price))
                for s in SYMBOLS:
                    await self.bot_place_order_safe(s, qty, xchange_client.Side.SELL, round(fair_price[s]))
                await self.place_swap_order("toAKAV", qty)
            self.log(f"Arbitrage opportunity: ETF {'over' if diff > 0 else 'under'}priced by {diff:.2f}")

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        # order = self.open_orders[order_id]
        # print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")
        pass

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        # print("order fill", self.positions)
        pass

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        # print("order rejected because of ", reason)
        pass


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
                
                fair_price_APT = None #earnings * constant_PE_ratio
                
                print(f"[{timestamp}] APT Earnings Update: {earnings}")
            elif subtype == "petition" and asset == "DLR":
                # Process DLR signature update news.
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]
                print(f"[{timestamp}] DLR: new signatures: {new_signatures}, cumulative: {cumulative}")
                self.dlr_cumulative = cumulative
                # Increment our event counter each time we receive a DLR signature update.
                self.dlr_event_count += 1
                # Update our fair price estimate using our refined method.
                self.update_dlr_fair_price(new_signatures, cumulative, timestamp)
            else:
                print(f"[{timestamp}] Received structured news for asset {asset} with subtype {subtype}")
        else:
            content = news_data.get("content", "")
            self.mkj_news_events.append((timestamp, content))
            fair_price_MKJ = None 
            #print(f"[{timestamp}] MKJ Unstructured News: {content}")
            #Carina still working on more advanded stuffs (only updated the certain parts to the git)





    def update_lognormal_params(self, new_signatures: int):
        """
        Update the estimates of ln_mu and ln_sigma using a rolling window of observations.
        For simplicity, we append the log(new_signatures) and compute the sample mean and std.
        """
        # Only update if new_signatures is positive
        if new_signatures > 0:
            log_val = math.log(new_signatures)
            self.log_increments.append(log_val)
            
            # Optionally, limit the size of the window to the most recent N events
            window_size = 50  # you can adjust this value
            if len(self.log_increments) > window_size:
                self.log_increments = self.log_increments[-window_size:]
            
            # Update parameters using the current window
            self.ln_mu = np.mean(self.log_increments)
            self.ln_sigma = np.std(self.log_increments, ddof=1)
            print(f"Updated ln_mu: {self.ln_mu:.4f}, ln_sigma: {self.ln_sigma:.4f}")


    def update_dlr_fair_price(self, new_signatures: int, cumulative: int, timestamp: int):
        """
        Update the fair price of DLR using the updated lognormal parameters.
        """
        # First, update our lognormal parameters with the new increment.
        self.update_lognormal_params(new_signatures)
        
        # Define the threshold for a successful outcome.
        threshold = 100000
        required = max(threshold - cumulative, 0)
        remaining_events = self.total_events_per_round - self.dlr_event_count
        
        if required == 0:
            probability = 1.0
        elif remaining_events <= 0:
            probability = 0.0
        else:
            # Mean and variance for one event based on the updated parameters.
            single_mean = math.exp(self.ln_mu + self.ln_sigma**2 / 2)
            single_var = (math.exp(self.ln_sigma**2) - 1) * math.exp(2 * self.ln_mu + self.ln_sigma**2)
            M = remaining_events * single_mean
            V = remaining_events * single_var

            # Fenton-Wilkinson approximation parameters for the sum.
            sigma_S_sq = math.log(1 + V / (M**2))
            mu_S = math.log(M) - sigma_S_sq / 2
            sigma_S = math.sqrt(sigma_S_sq)

            # Calculate the CDF for the lognormal approximation.
            try:
                ln_required = math.log(required)
                cdf = 0.5 * (1 + math.erf((ln_required - mu_S) / (sigma_S * math.sqrt(2))))
            except ValueError:
                cdf = 0.0

            # The probability of reaching the threshold is the chance that the sum of the remaining increments exceeds 'required'
            probability = 1 - cdf

        self.fair_price_dlr = 100 * probability
        print(f"[timestamp: {timestamp}] Updated DLR fair price: {self.fair_price_dlr:.2f} "
              f"(Probability: {probability:.2%}, remaining events: {remaining_events})")






    async def trade(self):
        while True:
            # TODO: replace mid-price with actual fair_price
            fair_price = {s: self.get_mid_price(s) for s in ALL_SYMBOLS}

            # ETF Arbitrage
            await self.bot_place_arbitrage_order(fair_price)

            # Market Making with Fade and Edge
            for symbol in ALL_SYMBOLS:
                position = self.positions[symbol]
                fade_adj = -self.fade * (1 if position > 0 else -1) * math.log2(1 + abs(position) / MAX_ABSOLUTE_POSITION)

                for side in [xchange_client.Side.BUY, xchange_client.Side.SELL]:
                    activity = self.get_market_activity_level(symbol, side)
                    edge = max(int(round(self.min_margin + (self.slack / 2 * math.tanh(-4 * self.edge_sensitivity * activity + 2)))), 1)

                    base_price = fair_price[symbol] + fade_adj
                    price = round(base_price - edge) if side == xchange_client.Side.BUY else round(base_price + edge)
                    await self.bot_place_order_safe(symbol, self.order_size, side, price)

                # Level Orders
                for level in range(len(self.spreads)):
                    spread = self.spreads[level]
                    bid = round(fair_price[symbol] - self.fade - spread)
                    ask = round(fair_price[symbol] + self.fade + spread)
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


