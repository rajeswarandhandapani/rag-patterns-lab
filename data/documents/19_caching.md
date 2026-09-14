# Caching Standard

Services use cache-aside reads. Cache keys include tenant ID and schema version. Positive entries have a default five-minute TTL; negative entries expire after 30 seconds.

On writes, the owning service invalidates its cache after the database transaction commits. A cache outage must degrade to the source of truth with rate limiting; it must not make writes fail.

