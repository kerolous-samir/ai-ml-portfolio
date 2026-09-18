FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py .
COPY superkart_best_model.pkl .

# Expose port for the API
EXPOSE 7860

# Run using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
