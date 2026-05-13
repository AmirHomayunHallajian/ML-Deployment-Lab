# Monitoring

This demo logs each prediction to `logs/predictions.log` with timestamp, risk probability, predicted class, and risk label.

The `/monitoring/summary` endpoint returns:
- number of predictions made
- average predicted risk probability
- high-risk prediction count
- low-risk prediction count
- service start time

Production ML systems need monitoring for reliability, quality, and compliance.

Limitations of this demo:
- In-memory counters reset when the service restarts.
- Logging uses local files, not centralized observability tools.
