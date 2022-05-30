from celery import Celery
from scoring_service import ScoringService
from config import *

celery = Celery(__name__)
celery.conf.broker_url = 'redis://redis:6379/0'
celery.conf.result_backend = 'redis://redis:6379/0'

@celery.task
def make_prediction(niiFilePath):
    ScoringService.predict(niiFilePath)
    return True