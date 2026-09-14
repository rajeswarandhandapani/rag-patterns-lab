# Beacon Identity Service

Beacon issues access tokens used by Atlas. Access tokens last 15 minutes and refresh tokens last 12 hours. Services validate the token audience and tenant claim. Machine clients use the client-credentials grant.

## Key rotation

Signing keys rotate every 30 days. Beacon publishes current and previous public keys through its JWKS endpoint for a 48-hour overlap. Alert `ID-207` means a service cannot refresh the JWKS cache; operators should verify private DNS before restarting anything.

