# Harbor Shipping Service

Harbor creates shipments only after an `OrderConfirmed` event. Domestic standard delivery targets three to five business days. Expedited delivery targets one to two business days.

Carrier callbacks are authenticated with an HMAC signature in `X-Harbor-Signature`. Failed signature validation returns `SHP-4012`; rotate the shared secret only after confirming the carrier and Harbor configurations differ.

