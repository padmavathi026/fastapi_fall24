
# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements
COPY ./requirements_final.txt /code/requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./diabetes_012_health_indicators_BRFSS2021.csv /code/
COPY ./generate_model.py /code/generate_model.py
COPY ./app /code/app

RUN python generate_model.py
RUN ls /code/app/

COPY ./saved_models/tuned_xgb_low_bmi.pkl /code/app/


RUN pip install xgboost


#RUN mv /code/tuned_xgb_low_bmi.pkl /code/app/tuned_xgb_low_bmi.pkl

#RUN rm/code/diabetes_012_health_indicators_BRFSS2021.csv


CMD ["fastapi", "run", "app/main.py", "--port", "80"]





