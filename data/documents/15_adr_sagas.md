# ADR-009: Checkout Saga

Checkout uses orchestration in Comet. Comet requests a Grove reservation, requests Flux authorization, then confirms both. If payment authorization fails, Comet releases inventory. If inventory confirmation fails after payment authorization, Comet voids the authorization and sends the order to manual review.

Every command and compensation is idempotent. Saga state is persisted before commands are emitted through the outbox.

