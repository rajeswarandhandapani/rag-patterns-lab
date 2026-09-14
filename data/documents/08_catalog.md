# Juniper Catalog Service

Juniper owns product descriptions, prices, and categories. Product updates reach the read cache within 60 seconds. Search documents update asynchronously within five minutes.

Price changes require an effective timestamp and a reason code. A request more than 10 minutes in the past fails with `CAT-4228`. Emergency corrections use reason `PRICE_CORRECTION` and require the pricing-admin role.

