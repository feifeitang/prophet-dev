try:
    import unzip_requirements
except ImportError:
    pass

import os
try:
    os.chmod("/tmp/sls-py-req/prophet/stan_model/prophet_model.bin", 0o777)
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def find_outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    outliers = df[((df < (q1-1.5*IQR)) | (df > (q3+1.5*IQR)))]
    return outliers


def mean_absolute_percent_error(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    return np.mean(np.abs((y - y_pred) / y)) * 100


def run(event, context):
    current_time = datetime.datetime.now().time()
    name = context.function_name

    body = {
        'message': 'Your cron function ' + name + ' ran at ' + str(current_time),
        'input': event
    }

    # load the data
    timestamps = get_metric_data()['MetricDataResults'][0]['Timestamps']
    values = get_metric_data()['MetricDataResults'][0]['Values']
    data = {
        'timestamps': timestamps,
        'values': values
    }
    df = pd.DataFrame(data)

    # find outliers
    outliers = find_outliers_IQR(df['values'])
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
    future = m.make_future_dataframe(periods=2016, freq="5min")
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    response = {
        'statusCode': 200,
        'body': json.dumps(body)
    }

    return response
