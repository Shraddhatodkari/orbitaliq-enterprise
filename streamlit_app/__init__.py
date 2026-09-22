"""OrbitalIQ Streamlit analyst dashboard.

A presentation layer over the existing FastAPI backend (``src/orbitaliq``).
This package never computes a financial metric, momentum score, satellite
change signal, or narrative — it only calls the backend's existing REST
API (see ``api_client.py``) and renders whatever the API returns. Every
number on screen traces back to a value the backend already computed and
disclosed a source for; every "Insufficient Data" / "Not Available" label
comes straight from the API response, never invented here.
"""
