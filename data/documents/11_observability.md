# Nova Observability Standard

Every request carries the W3C `traceparent` header and the Atlas correlation ID. Logs are structured JSON with tenant, service, severity, trace ID, and event name. Secrets, tokens, and full payment data are prohibited in logs.

Service-level objectives use a 30-day rolling window. Page when both the five-minute and one-hour error-budget burn rates exceed their thresholds; a single short spike creates a ticket rather than a page.

