#!/bin/bash

mkdir -p logs
# Start FastAPI in the background
# Start Streamlit
streamlit run upload_pdf.py --server.port 9000 > logs/streamlit.log 2>&1 &

python3 fastapi_app.py