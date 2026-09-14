# Atlas API Gateway

The Atlas gateway is the public entry point for all Northstar services. It authenticates OAuth 2.0 bearer tokens, assigns a correlation ID, and forwards requests to internal services. Clients must send `X-Tenant-ID`; missing tenant context produces `GW-104` with HTTP 400.

## Limits and timeouts

The default tenant limit is 600 requests per minute. A throttled request returns HTTP 429 and `GW-429`. Atlas waits 8 seconds for an upstream response before returning HTTP 504 and `GW-508`. Retries are the caller's responsibility and must use exponential backoff with jitter.

