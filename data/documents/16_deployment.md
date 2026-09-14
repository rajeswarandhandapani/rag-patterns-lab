# Deployment Runbook

Production deploys use a 5% canary for 15 minutes, followed by 25%, 50%, and 100% stages. Promotion requires healthy latency, error rate, saturation, and business success metrics. Database migrations must be backward compatible during the entire rollout.

Rollback the application before reversing data changes. Destructive migrations run only after old application versions and rollback windows have expired.

