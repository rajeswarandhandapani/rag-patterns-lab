# Incident INC-2025-031: Token Validation Failures

On 2025-04-03, three services rejected valid Beacon tokens with HTTP 401. The services could not resolve the private DNS name for Beacon's JWKS endpoint after a DNS zone link was removed.

Restarting pods did not help. Restoring the virtual-network link fixed resolution, and cached keys recovered automatically. The follow-up added synthetic JWKS resolution checks from every cluster.

