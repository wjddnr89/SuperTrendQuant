from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from supertrend_quant.brokers import TossBroker
from supertrend_quant.portfolio import OrderIntent


def response(payload=None, status_code: int = 200) -> Mock:
    result = Mock()
    result.status_code = status_code
    result.json.return_value = payload or {}
    result.raise_for_status.return_value = None
    return result


class TossBrokerContractAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.broker = TossBroker()
        self.token_patch = patch.object(self.broker, "_token", return_value="test-token")
        self.token_patch.start()

    def tearDown(self):
        self.token_patch.stop()

    @patch("supertrend_quant.brokers.requests.get")
    def test_account_uses_cash_buying_power_and_official_holding_fields(self, get):
        get.side_effect = [
            response({"result": {"currency": "USD", "cashBuyingPower": "3500.50"}}),
            response(
                {
                    "result": {
                        "items": [
                            {
                                "symbol": "AAPL",
                                "currency": "USD",
                                "quantity": "2.5",
                                "averagePurchasePrice": "100.25",
                                "lastPrice": "101.00",
                                "marketValue": {"amount": "252.50"},
                            },
                            {
                                "symbol": "005930",
                                "currency": "KRW",
                                "quantity": "10",
                                "averagePurchasePrice": "65000",
                                "marketValue": {"amount": "700000"},
                            },
                        ]
                    }
                }
            ),
        ]

        account = self.broker.get_account("US")

        self.assertEqual(account.cash, 3500.50)
        self.assertEqual(set(account.positions), {"AAPL"})
        self.assertEqual(account.positions["AAPL"].quantity, 2.5)
        self.assertEqual(account.positions["AAPL"].avg_price, 100.25)
        self.assertEqual(account.total_asset_value, 3753.0)
        self.assertIsNotNone(account.position_economics["AAPL"].net_return_pct)
        self.assertTrue(get.call_args_list[0].args[0].endswith("/api/v1/buying-power"))
        self.assertEqual(get.call_args_list[0].kwargs["params"], {"currency": "USD"})
        self.assertTrue(get.call_args_list[1].args[0].endswith("/api/v1/holdings"))

    @patch("supertrend_quant.brokers.requests.get")
    def test_prices_and_open_orders_use_documented_result_shapes(self, get):
        get.side_effect = [
            response({"result": [{"symbol": "AAPL", "lastPrice": "187.35"}]}),
            response(
                {
                    "result": {
                        "orders": [
                            {
                                "orderId": "order-1",
                                "symbol": "AAPL",
                                "side": "BUY",
                                "status": "PENDING",
                            }
                        ],
                        "nextCursor": None,
                        "hasNext": False,
                    }
                }
            ),
        ]

        self.assertEqual(self.broker.get_prices(["AAPL"]), {"AAPL": 187.35})
        orders = self.broker.list_open_orders()

        self.assertEqual(orders[0]["orderId"], "order-1")
        order_call = get.call_args_list[1]
        self.assertTrue(order_call.args[0].endswith("/api/v1/orders"))
        self.assertEqual(order_call.kwargs["params"], {"status": "OPEN"})

    @patch("supertrend_quant.brokers.requests.get")
    def test_prices_are_split_at_the_official_two_hundred_symbol_limit(self, get):
        get.side_effect = [
            response({"result": [{"symbol": "S000", "lastPrice": "1"}]}),
            response({"result": [{"symbol": "S200", "lastPrice": "2"}]}),
        ]

        prices = self.broker.get_prices([f"S{index:03d}" for index in range(201)])

        self.assertEqual(prices, {"S000": 1.0, "S200": 2.0})
        self.assertEqual(get.call_count, 2)
        self.assertEqual(
            len(get.call_args_list[0].kwargs["params"]["symbols"].split(",")),
            200,
        )
        self.assertEqual(
            len(get.call_args_list[1].kwargs["params"]["symbols"].split(",")),
            1,
        )

    @patch("supertrend_quant.brokers.requests.get")
    def test_order_detail_exposes_nested_execution_for_reconciliation(self, get):
        get.return_value = response(
            {
                "result": {
                    "orderId": "order-1",
                    "status": "FILLED",
                    "execution": {"filledQuantity": "3"},
                }
            }
        )

        detail = self.broker.get_order("order-1")

        self.assertEqual(detail["status"], "FILLED")
        self.assertTrue(get.call_args.args[0].endswith("/api/v1/orders/order-1"))

    @patch("supertrend_quant.brokers.requests.post")
    def test_cancel_uses_post_cancel_endpoint(self, post):
        post.return_value = response({"result": {"orderId": "order-1"}}, status_code=200)

        self.assertTrue(self.broker.cancel_order("order-1"))

        call = post.call_args
        self.assertTrue(call.args[0].endswith("/api/v1/orders/order-1/cancel"))
        self.assertEqual(call.kwargs["json"], {})

    @patch("supertrend_quant.brokers.requests.post")
    def test_us_limit_order_preserves_decimal_price(self, post):
        post.return_value = response({"result": {"orderId": "order-2"}}, status_code=201)
        order = OrderIntent(
            symbol="AAPL",
            side="buy",
            quantity=2,
            order_type="limit",
            price=Decimal("185.75"),
        )

        self.assertTrue(self.broker.place_order(order))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload,
            {
                "symbol": "AAPL",
                "side": "BUY",
                "orderType": "LIMIT",
                "quantity": "2",
                "price": "185.75",
            },
        )

    @patch("supertrend_quant.brokers.requests.post")
    def test_oauth_uses_required_form_fields_instead_of_basic_auth(self, post):
        broker = TossBroker()
        broker.client_id = "client-id"
        broker.client_secret = "client-secret"
        broker.token = None
        post.return_value = response(
            {"access_token": "token", "expires_in": 3600}
        )

        token = broker._token()

        self.assertEqual(token, "token")
        self.assertEqual(
            post.call_args.kwargs["data"],
            {
                "grant_type": "client_credentials",
                "client_id": "client-id",
                "client_secret": "client-secret",
            },
        )
        self.assertNotIn("auth", post.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
