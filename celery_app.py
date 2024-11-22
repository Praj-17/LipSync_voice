# celery_app.py

from celery import Celery

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Redis broker URL
    backend='redis://localhost:6379/0'  # Redis backend URL
)

celery_app.conf.update(
    result_expires=3600,  # Task results expiration time in seconds
)
