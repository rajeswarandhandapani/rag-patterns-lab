# Meridian Event Platform

Meridian is the shared Kafka platform. Producers use the outbox pattern when changing a database and publishing a domain event. Consumers commit offsets only after their local transaction succeeds.

Events remain for seven days. Schema compatibility is backward by default. A breaking schema change requires a new topic version and a migration period with dual publishing.

