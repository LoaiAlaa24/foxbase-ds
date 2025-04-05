#!/bin/bash

# Function to handle cleanup on interrupt
cleanup() {
    echo "Cleaning up... Stopping FastAPI and Streamlit apps."

    # Check if FastAPI process is still running before attempting to kill it
    if ps -p $FASTAPI_PID > /dev/null; then
        echo "Stopping FastAPI..."
        kill $FASTAPI_PID
    else
        echo "FastAPI process not found."
    fi

    # Check if Streamlit process is still running before attempting to kill it
    if ps -p $STREAMLIT_PID > /dev/null; then
        echo "Stopping Streamlit..."
        kill $STREAMLIT_PID
    else
        echo "Streamlit process not found."
    fi

    exit 0
}

# Trap SIGINT (Ctrl+C) signal to call cleanup
trap cleanup SIGINT

# Start FastAPI app
echo "Starting FastAPI app..."
uvicorn main:app --reload &
FASTAPI_PID=$!
echo "FastAPI started with PID $FASTAPI_PID."

# # Wait for FastAPI to initialize
# sleep 5

# # Start Streamlit app
# echo "Starting Streamlit app..."
# streamlit run streamlit_chatbot.py &
# STREAMLIT_PID=$!
# echo "Streamlit started with PID $STREAMLIT_PID."

# Wait for processes to finish
wait $FASTAPI_PID
wait $STREAMLIT_PID
