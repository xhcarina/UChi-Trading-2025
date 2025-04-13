from collections import defaultdict

def find_fair_price(orders):
    """
    Given a list of orders, each of which is (price, volume, order_type),
    where order_type is either 'bid' or 'ask',
    return the 'fair price' according to the parentheses-balancing idea.
    """
    # 1. Aggregate volumes by price
    net_volumes = defaultdict(int)
    for price, volume, order_type in orders:
        if order_type.lower() == 'bid':
            net_volumes[price] += volume   # bids add positive volume
        else:
            net_volumes[price] -= volume   # asks add negative volume

    # 2. Sort prices in ascending order
    sorted_prices = sorted(net_volumes.keys())

    # 3. Build a list of parentheses and remember each price's index range
    parentheses = []
    price_index_ranges = []  # will hold tuples (price, start_index, end_index)
    current_index = 0

    for p in sorted_prices:
        v = net_volumes[p]
        count = abs(v)
        start_idx = current_index
        if v > 0:
            # Add "(" v times
            parentheses.extend(["("] * count)
        elif v < 0:
            # Add ")" abs(v) times
            parentheses.extend([")"] * count)
        end_idx = current_index + count - 1
        current_index += count

        # Record which chunk of parentheses corresponds to this price
        if count > 0:  # only store if there's actual volume
            price_index_ranges.append((p, start_idx, end_idx))

    # If there's no volume at all, just return None or something sensible
    if not parentheses:
        return None
    
    # 4. Compute prefix of "(" counts
    prefix_open = [0] * (len(parentheses) + 1)
    for i in range(len(parentheses)):
        prefix_open[i+1] = prefix_open[i] + (1 if parentheses[i] == "(" else 0)



    # Compute suffix of ")" counts
    suffix_close = [0] * (len(parentheses) + 1)
    for i in range(len(parentheses) - 1, -1, -1):
        suffix_close[i] = suffix_close[i+1] + (1 if parentheses[i] == ")" else 0)

    # 5. Find index i that minimizes |O(i) - C(i)|
    best_i = 0
    best_diff = float('inf')
    n = len(parentheses)

    for i in range(n + 1):
        # O(i) = # of "(" up to index i
        Oi = prefix_open[i]
        # C(i) = # of ")" from index i to the end
        Ci = suffix_close[i]
        diff = abs(Oi - Ci)
        if diff < best_diff:
            best_diff = diff
            best_i = i

    # 6. Map best_i back to a price
    # best_i is effectively "between" best_i-1 and best_i in the parentheses array.
    # We'll say that if best_i falls within a price's index range, that price is "fair."
    fair_price = None
    if best_i == 0:
        # If best_i == 0, we are at the very beginning,
        # so pick the first price in the sorted list (if any)
        if price_index_ranges:
            fair_price = price_index_ranges[0][0]
    elif best_i == n:
        # If best_i == n, we are at the very end,
        # so pick the last price in the sorted list
        if price_index_ranges:
            fair_price = price_index_ranges[-1][0]
    else:
        # Otherwise, find the chunk that covers best_i
        for p, start_idx, end_idx in price_index_ranges:
            if start_idx <= best_i <= end_idx:
                fair_price = p
                break

    return fair_price

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Suppose we have a simple order book:
    # price=100, bid volume=3
    # price=101, ask volume=2
    # price=102, bid volume=5
    # price=99, ask volume=4
    sample_orders = [
        (100, 3, 'bid'),
        (101, 2, 'ask'),
        (102, 5, 'bid'),
        (99,  4, 'ask'),
    ]

    price = find_fair_price(sample_orders)
    print("Fair price is:", price)

