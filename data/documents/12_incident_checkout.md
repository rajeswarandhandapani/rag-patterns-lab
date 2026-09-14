# Incident INC-2025-017: Checkout Latency

On 2025-02-14, checkout p95 latency rose from 480 ms to 9.2 seconds. Flux was healthy. Grove's connection pool was exhausted after a deployment changed the pool size from 80 to 8.

Operators rolled back Grove release `2025.02.14.3`, restoring latency in six minutes. The permanent actions were a minimum pool-size guardrail, a deployment canary, and a dashboard comparing active requests with available connections.

