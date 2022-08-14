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

BUCKET = 'fb-production-metric'
ORG = 'a83f3349e41bdab9'
TOKEN = '3zdSyqF8QcIFM0PHymCYDltYFZq1lg5JI6AgtMtGfk6AF7i9krWnvdIDWaUnLhbDUWJwZiGme40c5JURo8zduw=='
URL = 'https://us-west-2-2.aws.cloud2.influxdata.com/'


def query_influxdb():
    try:
        client = influxdb_client.InfluxDBClient(
            url=URL,
            token=TOKEN,
            org=ORG
        )

        query_api = client.query_api()

        query = ' from(bucket:"fb-production-metric")\
        |> range(start: -28d)\
        |> filter(fn:(r) => r._measurement == "ras")\
        |> filter(fn:(r) => r.client_country == "US")\
        |> filter(fn:(r) => r.pid == "TMCHECKM")\
        |> filter(fn:(r) => r._field == "check_telephone")\
        |> aggregateWindow(every: 1h, fn: count)\
        |> yield() '

        result = query_api.query(org=ORG, query=query)

        times = []
        values = []
        for table in result:
            for record in table.records:
                times.append((record.get_time()))
                values.append((record.get_value()))

        return {
            'times': times,
            'values': values
        }

    except Exception as e:
        print('query_influxdb error:', str(e))


def find_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    outliers = df[((df < (q1-1.5*IQR)) | (df > (q3+1.5*IQR)))]
    return outliers


def calculate_mape(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    return np.mean(np.abs((y - y_pred) / y)) * 100


def prophet_forecast(df):
    try:
        # model fitting
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=24, freq='1h')
        forecast = m.predict(future)
        print('----- forecast dataframe start -----')
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        print('----- forecast dataframe end -----')

        # cross validation
        df_cv = cross_validation(
            m, initial='3 days', period='1 days', horizon='2 days')
        mape = calculate_mape(df_cv['y'], df_cv['yhat'])
        print('mape', mape)
        print('acc', 100-mape)

        return forecast

    except Exception as e:
        print('prophet_forecast error:', str(e))


def create_write_influxdb_data(df, forecast):
    df_concat = pd.concat([df[['ds', 'y']], forecast[['yhat', 'yhat_lower', 'yhat_upper']]], axis = 1)

    ds = df_concat.ds.values

    # y
    values = df_concat.head(len(df)).y.values
    variables = ['y'] * len(values)
    data = {
        'ds': df_concat.head(len(df)).ds.values,
        'check_telephone_count': values,
        'variable': variables
    }
    y_df = pd.DataFrame(data)
    y_df.set_index('ds', inplace=True)

    # yhat
    values = df_concat.yhat.values
    variables = ['yhat'] * len(values)
    data = {
        'ds': ds,
        'check_telephone_count': values,
        'variable': variables
    }
    yhat_df = pd.DataFrame(data)
    yhat_df.set_index('ds', inplace=True)

    # yhat_lower
    values = df_concat.yhat_lower.values
    variables = ['yhat_lower'] * len(values)
    data = {
        'ds': ds,
        'check_telephone_count': values,
        'variable': variables
    }
    yhat_lower_df = pd.DataFrame(data)
    yhat_lower_df.set_index('ds', inplace=True)

    # yhat_upper
    values = df_concat.yhat_upper.values
    variables = ['yhat_upper'] * len(values)
    data = {
        'ds': ds,
        'check_telephone_count': values,
        'variable': variables
    }
    yhat_upper_df = pd.DataFrame(data)
    yhat_upper_df.set_index('ds', inplace=True)

    return {
        'y': y_df,
        'yhat': yhat_df,
        'yhat_lower': yhat_lower_df,
        'yhat_upper': yhat_upper_df
    }


def write_influxdb(data):
    try:
        client = influxdb_client.InfluxDBClient(
            url=URL,
            token=TOKEN,
            org=ORG
        )

        write_api = client.write_api(write_options=SYNCHRONOUS)

        y_df = data['y']
        yhat_df = data['yhat']
        yhat_lower_df = data['yhat_lower']
        yhat_upper_df = data['yhat_upper']

        write_api.write(bucket=BUCKET, org=ORG, record=y_df, data_frame_measurement_name='ras-prophet-forecast', data_frame_tag_columns=['variable'])
        write_api.write(bucket=BUCKET, org=ORG, record=yhat_df, data_frame_measurement_name='ras-prophet-forecast', data_frame_tag_columns=['variable'])
        write_api.write(bucket=BUCKET, org=ORG, record=yhat_lower_df, data_frame_measurement_name='ras-prophet-forecast', data_frame_tag_columns=['variable'])
        write_api.write(bucket=BUCKET, org=ORG, record=yhat_upper_df, data_frame_measurement_name='ras-prophet-forecast', data_frame_tag_columns=['variable'])

        client.close()

    except Exception as e:
        print('write_influxdb error:', str(e))


def run(event, context):
    current_time = datetime.datetime.now().time()
    name = context.function_name

    body = {
        'message': 'Your cron function ' + name + ' ran at ' + str(current_time),
        'input': event
    }

    # load the data
    query_influxdb_res = query_influxdb()
    times = query_influxdb_res['times']
    values = query_influxdb_res['values']
    data = {
        'times': times,
        'values': values
    }
    df = pd.DataFrame(data)
    print(df)

    # find outliers
    outliers = find_outliers(df['values'])
    print('number of outliers: ' + str(len(outliers)))
    print('percentage of extreme observations: ' +
          str(round(len(outliers)/len(df)*100, 4)) + '%')
    print('max outlier value: ' + str(outliers.max()))
    print('min outlier value: ' + str(outliers.min()))

    # data preprocessing
    data = {
        'ds': df['times'].values,
        'y': df['values'].values
    }
    df = pd.DataFrame(data)

    print('---------- prophet forecast start ----------')
    forecast = prophet_forecast(df)
    print('---------- prophet forecast end ----------')

    data = create_write_influxdb_data(df, forecast)
    
    write_influxdb(data)

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
