# Security and Data Handling

Customer contact details are confidential and encrypted in transit and at rest. Payment card numbers never enter Northstar services; Flux stores only processor tokens and the last four digits.

Production access is just in time, expires after one hour, and requires a ticket. Tenant data access is audited. Support exports expire after 24 hours and must use the approved encrypted container.

