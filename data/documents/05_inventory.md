# Grove Inventory Service

Grove tracks available, reserved, and committed stock. A reservation lasts 20 minutes. Confirmation from Comet commits the stock; expiration releases it.

The write API uses optimistic concurrency through an `ETag`. A stale `If-Match` header returns HTTP 412 and `INV-412`. The client should reload the item, reapply its change, and retry with the new ETag.

