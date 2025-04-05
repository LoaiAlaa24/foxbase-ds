# Use the official Python image
FROM python:3.13.0

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY ui_interface/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY ui_interface .

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
