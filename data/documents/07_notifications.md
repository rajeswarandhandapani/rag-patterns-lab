# Iris Notification Service

Iris sends email, SMS, and push messages from domain events. Delivery is at least once, so templates and downstream providers must tolerate duplicates. Iris deduplicates on event ID for 24 hours.

Messages enter the dead-letter queue after five failed delivery attempts. Operators replay the queue in batches of 100 after fixing the provider or template issue. Never replay while the provider outage is active.

