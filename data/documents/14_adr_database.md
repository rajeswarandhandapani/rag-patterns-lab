# ADR-004: Database Ownership

Each service owns its database and schema. Direct cross-service database reads are forbidden because they couple deployments and bypass service authorization. Synchronous reads use the owning service API; asynchronous views use published domain events.

Analytics receives change-data-capture feeds in a separate warehouse. Analytics access does not permit operational services to query another service's tables.

