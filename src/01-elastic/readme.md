# Elastic Playground


### Install requirements
``pip install -r requirements.txt``

### Run
Ensure to create `.env` file in the directory and put elastic host uri and api-key.

``uvicorn app:app --reload --env-file .env``
