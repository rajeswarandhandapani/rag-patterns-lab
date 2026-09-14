# Kite Search Service

Kite provides customer-facing product search. It consumes catalog changes from Juniper and maintains a denormalized index. Results may therefore lag catalog writes by up to five minutes.

The `/v2/search` endpoint supports filters for category, availability, and price range. Queries are capped at 200 characters. Error `SRC-1009` indicates an invalid filter expression.

