# filename: ib_executor.py

"""
Interactive Brokers Order Executor

Responsible for:
- Receiving strategy decisions
- Creating IB orders
- Sending orders through IB API

Flow:

strategy.py
      |
      v
ib_executor.py
      |
      v
IBClient
      |
      v
TWS / IB Gateway


Current version:
- Market entry orders
- Stop loss orders
- Take profit orders
- Basic order tracking

"""


from ibapi.order import Order

from ib_client import IBClient




# -----------------------------------------
# Order Executor
# -----------------------------------------

class IBExecutor:



    def __init__(self, ib_client: IBClient):

        self.ib = ib_client

        self.order_id = None


        self.active_orders = {}



    # -------------------------------------
    # Generate next order id
    # -------------------------------------

    def get_order_id(self):


        if self.order_id is None:

            self.order_id = (
                self.ib.next_order_id
            )


        current_id = self.order_id


        self.order_id += 1


        return current_id



    # -------------------------------------
    # Create market order
    # -------------------------------------

    def create_market_order(
        self,
        action,
        quantity
    ):


        order = Order()


        order.action = action


        order.orderType = "MKT"


        order.totalQuantity = quantity


        order.transmit = True


        return order




    # -------------------------------------
    # Create stop order
    # -------------------------------------

    def create_stop_order(
        self,
        action,
        quantity,
        stop_price
    ):


        order = Order()


        order.action = action


        order.orderType = "STP"


        order.totalQuantity = quantity


        order.auxPrice = round(
            stop_price,
            2
        )


        order.transmit = True


        return order




    # -------------------------------------
    # Create limit order
    # -------------------------------------

    def create_take_profit_order(
        self,
        action,
        quantity,
        limit_price
    ):


        order = Order()


        order.action = action


        order.orderType = "LMT"


        order.totalQuantity = quantity


        order.lmtPrice = round(
            limit_price,
            2
        )


        order.transmit = True


        return order




    # -------------------------------------
    # Execute strategy decision
    # -------------------------------------

    def execute(
        self,
        contract,
        decision
    ):



        action = decision.get(
            "action"
        )


        if action == "HOLD":


            print(
                "No order generated"
            )


            return None




        side = decision.get(
            "side"
        )


        quantity = decision.get(
            "quantity"
        )


        stop_loss = decision.get(
            "stop_loss"
        )


        take_profit = decision.get(
            "take_profit"
        )



        print(
            f"""
Executing Order

Action:
{action}

Side:
{side}

Quantity:
{quantity}

Stop:
{stop_loss}

Target:
{take_profit}

"""
        )



        # ------------------------------
        # Entry order
        # ------------------------------

        entry_id = self.get_order_id()



        entry_order = self.create_market_order(

            side,

            quantity

        )



        self.ib.placeOrder(

            entry_id,

            contract,

            entry_order

        )



        self.active_orders[entry_id] = {

            "type":
                "ENTRY",

            "side":
                side,

            "quantity":
                quantity

        }



        # ------------------------------
        # Protective orders
        # ------------------------------

        if side == "BUY":


            stop_side = "SELL"

            target_side = "SELL"


        else:


            stop_side = "BUY"

            target_side = "BUY"




        # Stop loss

        stop_id = self.get_order_id()


        stop_order = self.create_stop_order(

            stop_side,

            quantity,

            stop_loss

        )


        self.ib.placeOrder(

            stop_id,

            contract,

            stop_order

        )




        # Take profit

        target_id = self.get_order_id()


        target_order = self.create_take_profit_order(

            target_side,

            quantity,

            take_profit

        )


        self.ib.placeOrder(

            target_id,

            contract,

            target_order

        )



        print(
            "Orders submitted successfully"
        )


        return {


            "entry_order_id":
                entry_id,


            "stop_order_id":
                stop_id,


            "target_order_id":
                target_id


        }






# -----------------------------------------
# Example usage
# -----------------------------------------

if __name__ == "__main__":


    from ib_client import IBClient



    ib = IBClient()


    ib.connect_ib()



    executor = IBExecutor(
        ib
    )



    contract = ib.stock_contract(
        "IONQ"
    )



    example_decision = {


        "action":
            "OPEN",


        "side":
            "BUY",


        "quantity":
            100,


        "stop_loss":
            30.50,


        "take_profit":
            34.00

    }



    executor.execute(

        contract,

        example_decision

    )