try:
    import unzip_requirements
except ImportError:
    pass

import os
try:
    os.chmod('/tmp/sls-py-req/prophet/stan_model/prophet_model.bin', 0o777)
    print('----------------')
    print('set permission succ')
    print('----------------')
except Exception as e:
    print('set permission error:', str(e))

import json
import boto3
import datetime
import logging
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

bucket = 'fb-production-metric'
org = 'a83f3349e41bdab9'
token = '3zdSyqF8QcIFM0PHymCYDltYFZq1lg5JI6AgtMtGfk6AF7i9krWnvdIDWaUnLhbDUWJwZiGme40c5JURo8zduw=='
url = 'https://us-west-2-2.aws.cloud2.influxdata.com/'


def query_influxdb():
    try:
        client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org
        )

        query_api = client.query_api()

        query = ' from(bucket:"fb-production-metric")\
        |> range(start: -28d)\
        |> filter(fn:(r) => r._measurement == "ras")\
        |> filter(fn:(r) => r.client_country == "US")\
        |> filter(fn:(r) => r.pid == "TMCHECKM")\
        |> aggregateWindow(every: 1h, fn: count)\
        |> yield() '

        result = query_api.query(org=org, query=query)

        results = []
        for table in result:
            for record in table.records:
                results.append((record.get_field(), record.get_value()))

        print(results)

    except Exception as e:
        print('queryInfluxCloud error:', str(e))


def find_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    outliers = df[((df < (q1-1.5*IQR)) | (df > (q3+1.5*IQR)))]
    return outliers


def calculate_mape(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    return np.mean(np.abs((y - y_pred) / y)) * 100


def prophet_forecast():
    try:
        # load the data
        timestamps = get_metric_data()['MetricDataResults'][0]['Timestamps']
        values = get_metric_data()['MetricDataResults'][0]['Values']
        data = {
            'timestamps': timestamps,
            'values': values
        }
        df = pd.DataFrame(data)

        # find outliers
        outliers = find_outliers(df['values'])
        print('number of outliers: ' + str(len(outliers)))
        print('percentage of extreme observations: ' +
              str(round(len(outliers)/len(df)*100, 4)) + '%')
        print('max outlier value: ' + str(outliers.max()))
        print('min outlier value: ' + str(outliers.min()))

        # data preprocessing
        data = {
            'ds': df['timestamps'].values,
            'y': df['values'].values
        }
        df = pd.DataFrame(data)

        # model fitting
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=2016, freq='5min')
        forecast = m.predict(future)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # cross validation
        df_cv = cross_validation(
            m, initial='8 days', period='1 days', horizon='2 days')
        mape = calculate_mape(df_cv['y'], df_cv['yhat'])
        print('mape', mape)
        print('acc', 100-mape)

    except Exception as e:
        print('prophet_forecast error:', str(e))


def write_influxdb(points):
    try:
        client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org
        )

        write_api = client.write_api(write_options=SYNCHRONOUS)

        # points = influxdb_client.Point("my_measurement").tag("location", "Prague").field("temperature", 25.3)
        write_api.write(bucket=bucket, org=org, record=points)

    except Exception as e:
        print('write_influxdb error:', str(e))


def run(event, context):
    current_time = datetime.datetime.now().time()
    name = context.function_name

    body = {
        'message': 'Your cron function ' + name + ' ran at ' + str(current_time),
        'input': event
    }

    query_influxdb()

    response = {
        'statusCode': 200,
        'body': json.dumps(body)
    }

    return response


def get_metric_data():
    try:
        client = boto3.client('cloudwatch')

        response = client.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'req',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/EC2',
                            'MetricName': 'CPUUtilization',
                            'Dimensions': [
                                {
                                    'Name': 'InstanceId',
                                    'Value': 'i-0fd789cbc1b080dd1'
                                },
                            ],
                        },
                        'Period': 300,
                        'Stat': 'Average',
                    },
                    'Label': 'reqLabel',
                    'ReturnData': True,
                },
            ],
            StartTime=datetime.datetime.now()-datetime.timedelta(28),
            EndTime=datetime.datetime.now(),
            ScanBy='TimestampAscending',
            LabelOptions={'Timezone': '+0800'},
        )

        return response

    except Exception as e:
        print('get_metric_data error:', str(e))
