# Disaster Recovery Plan

Tier-one services have a recovery time objective of 30 minutes and a recovery point objective of five minutes. Databases replicate to the paired region. Meridian events are copied to the recovery cluster with a target lag below two minutes.

Regional exercises occur twice a year. Traffic moves only after identity, data replication, and outbound dependencies pass automated checks.

