# Flux Payment Service

Flux authorizes and captures card payments. Comet sends an order ID as the payment idempotency key. Authorization expires after seven days if it is not captured.

Error `PAY-2031` means the processor timed out before returning a definitive result. Do not immediately retry a charge. Query `GET /v1/payments/{orderId}` first; retry only when the status is `NOT_FOUND`. Duplicate capture attempts return the original capture response.

