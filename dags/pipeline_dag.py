from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etl')))
from load_data import load_data
from preprocess import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from save_results import save_to_local_storage
import pandas as pd
import joblib

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    dag_id='ml_pipeline_breast_cancer',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    description='ML pipeline with Airflow for breast cancer diagnosis',
) as dag:

    def task_load_data(**kwargs):
        df = load_data('/opt/airflow/datasets/wdbc.data.csv')
        df.to_pickle('/opt/airflow/temp/df.pkl')

    def task_preprocess(**kwargs):
        df = pd.read_pickle('/opt/airflow/temp/df.pkl')
        X, y = preprocess_data(df)
        joblib.dump((X, y), '/opt/airflow/temp/data.pkl')

    def task_train_model(**kwargs):
        X, y = joblib.load('/opt/airflow/temp/data.pkl')
        train_model(X, y, model_path='/opt/airflow/results/model.pkl')

    def task_evaluate_model(**kwargs):
        model = joblib.load('/opt/airflow/results/model.pkl')
        X, y = joblib.load('/opt/airflow/temp/data.pkl')
        evaluate_model(model, X, y, metrics_path='/opt/airflow/results/metrics.json')

    def task_save_results(**kwargs):
        save_to_local_storage('/opt/airflow/results/metrics.json', 'results/')
        save_to_local_storage('/opt/airflow/results/model.pkl', 'results/')

    t1 = PythonOperator(task_id='load_data', python_callable=task_load_data)
    t2 = PythonOperator(task_id='preprocess_data', python_callable=task_preprocess)
    t3 = PythonOperator(task_id='train_model', python_callable=task_train_model)
    t4 = PythonOperator(task_id='evaluate_model', python_callable=task_evaluate_model)
    t5 = PythonOperator(task_id='save_results', python_callable=task_save_results)

    t1 >> t2 >> t3 >> t4 >> t5