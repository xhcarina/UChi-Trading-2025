from typing import Optional
import xchange_client
import asyncio
import argparse

class MyXchangeClient(xchange_client.XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        
        
        #Pair_price
        self.fair_price_APT = None    # Calculated fair price for APT based on earnings
        self.fair_price_DLR = None    # Calculated fair price for DLR based on signatures
        self.fair_price_MKJ = None    # Calculated fair price for MKJ based on unstructured news

         
        #Carina's news variable
        self.apt_earnings = None      # Latest earnings value for APT
        self.dlr_signatures = 0       # Cumulative petition signatures for DLR
        
        self.mkj_news_events = []     # List of unstructured news events for MKJ
        
    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("Order fill received. Updated positions:", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("Order rejected because of:", reason)

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
        # Optional: implement logic to react to changes in the order book.
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # Optional: implement additional actions upon receiving a swap response.
        pass

    async def bot_handle_news(self, news_release: dict):
        """
        Processes incoming news events from the exchange.
        
        For APT (earnings news):
          - Structured earnings news is received twice per day.
          - Use the earnings value to compute a new fair price.
          #fair_price_APT = <YOUR_FORMULA_HERE>
          
        For DLR (petition news):
          - Structured petition news is received five times per day.
          - The cumulative signatures are used to determine DLR's valuation.
          #fair_price_DLR = 100 if cumulative >= 100000 else 0
          
        For MKJ (unstructured news):
          - Unstructured news arrives randomly.
          - Use your quantitative model to determine MKJ's fair value.
          #fair_price_MKJ = <YOUR_MODEL_BASED_FORMULA>
        """
        timestamp = news_release["timestamp"]  # Exchange ticks (not ISO or Epoch)
        news_type = news_release["kind"]
        news_data = news_release["new_data"]

        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            asset = news_data["asset"]
            if subtype == "earnings" and asset == "APT":
                earnings = news_data["value"]
                self.apt_earnings = earnings
                
                fair_price_APT = #earnings * constant_PE_ratio
                
                print(f"[{timestamp}] APT Earnings Update: {earnings}")
            elif subtype == "petition" and asset == "DLR":
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]
                self.dlr_signatures = cumulative
                if cumulative >= 100000:
                    
                    fair_price_DLR = #something 
                else:
                    fair_price_DLR = 0
                
                print(f"[{timestamp}] DLR Petition Update: +{new_signatures} new, cumulative {cumulative}")
            else:
                print(f"[{timestamp}] Received structured news for asset {asset} with subtype {subtype}")
        else:
            content = news_data.get("content", "")
            self.mkj_news_events.append((timestamp, content))
            # Calculate MKJ's fair price using your chosen quantitative model:
            #fair_price_MKJ = <YOUR_MODEL_BASED_FORMULA>
            print(f"[{timestamp}] MKJ Unstructured News: {content}")



    async def trade(self):
        # Example trading sequence demonstrating various actions:
        await asyncio.sleep(5)
        print("Attempting to trade...")
        await self.place_order("APT", 3, xchange_client.Side.BUY, 5)
        await self.place_order("APT", 3, xchange_client.Side.SELL, 7)
        await asyncio.sleep(5)
        # Cancel the first open order as an example:
        await self.cancel_order(list(self.open_orders.keys())[0])
        await self.place_swap_order('toAKAV', 1)
        await asyncio.sleep(5)
        await self.place_swap_order('fromAKAV', 1)
        await asyncio.sleep(5)
        await self.place_order("APT", 1000, xchange_client.Side.SELL, 7)
        await asyncio.sleep(5)
        market_order_id = await self.place_order("APT", 10, xchange_client.Side.SELL)
        print("Market Order ID:", market_order_id)
        await asyncio.sleep(5)
        print("Current positions:", self.positions)

    async def view_books(self):
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k, v) for k, v in book.bids.items() if v != 0)
                sorted_asks = sorted((k, v) for k, v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self, user_interface):
        asyncio.create_task(self.trade())
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())
        await self.connect()


async def main(user_interface: bool):
    SERVER = 'SERVER URL'
    my_client = MyXchangeClient(SERVER, "USERNAME", "PASSWORD")
    await my_client.start(user_interface)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client connecting to exchange, running trading logic, and optionally deploying the Phoenixhood API."
    )
    parser.add_argument("--phoenixhood", required=False, default=False, type=bool,
                        help="Starts Phoenixhood API if true")
    args = parser.parse_args()
    user_interface = args.phoenixhood
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(user_interface))
