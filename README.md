# SuperResolutionDentalXray

### Docker Guide
docker build -t super-resolution-dental-xray .


docker run -d -p 8000:8000 super-resolution-dental-xray



### Test

pip install pytest
pytest


test reports
pip install pytest-cov
pytest --cov=src
