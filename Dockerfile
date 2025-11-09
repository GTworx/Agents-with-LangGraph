FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cooking_agent.py .

# Run the application
CMD ["python", "cooking_agent.py"]