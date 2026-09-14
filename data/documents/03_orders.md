# Comet Order Service

Comet owns order creation and order status. `POST /v1/orders` requires an idempotency key and returns the existing order when that key is replayed with an identical body. Reuse with a different body returns HTTP 409 and `ORD-4097`.

Orders move through `PENDING`, `CONFIRMED`, `FULFILLED`, or `CANCELLED`. Only pending or confirmed orders can be cancelled. Comet emits `OrderCreated`, `OrderConfirmed`, and `OrderCancelled` events.

