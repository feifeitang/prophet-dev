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

client = influxdb_client.InfluxDBClient(
    url=URL,
    token=TOKEN,
    org=ORG
)


def delete_data():
    try:
        delete_api = client.delete_api()

        start = datetime.datetime.now()
        print('start time', start)
        stop = datetime.datetime.now()+datetime.timedelta(5)
        print('stop time', stop)
        delete_api.delete(
            start, stop, '_measurement="ras-prophet-forecast"', bucket=BUCKET)

    except Exception as e:
        print('delete_data error:', str(e))


def create_query(measurement, tagKey, tagValue, interval):
    try:
        query = ' from(bucket:"{}")\
        |> range(start: -10d)\
        |> filter(fn: (r) => r._measurement == "{}" and r.{} == "{}")\
        |> group(columns: ["{}"])\
        |> aggregateWindow(every: {}, fn: count)\
        |> yield() '.format(BUCKET, measurement, tagKey, tagValue, tagKey, interval)

        return query

    except Exception as e:
        print('create_query error:', str(e))


def query_influxdb(query):
    try:
        query_api = client.query_api()

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
        # # Tune changepoint
        # params_grid = {
        #     'n_changepoints': [20, 25, 30],  # default 25
        #     'changepoint_range': [7, 8, 9],  # default 0.8
        #     'changepoint_prior_scale': [0.5, 0.05],  # default 0.05
        # }
        # grid = ParameterGrid(params_grid)
        # cnt = 0
        # for p in grid:
        #     cnt = cnt+1
        # print('Total Possible Models', cnt)

        # val = pd.DataFrame(columns = ['Acc', 'MAPE', 'Parameters'])

        # for p in grid:
        #     print(p)

        #     m = Prophet(
        #         n_changepoints = p['n_changepoints'],
        #         changepoint_range = p['changepoint_range'] / 10,
        #         changepoint_prior_scale = p['changepoint_prior_scale'],
        #     )
        #     m.fit(df)

        #     future = m.make_future_dataframe(periods=72, freq="1h")
        #     forecast = m.predict(future)
        #     forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        #     df_cv = cross_validation(m, initial='7 days', period='1.5 days', horizon = '3 days')

        #     mape = calculate_mape(df_cv['y'], df_cv['yhat'])
        #     acc = 100-mape
        #     val = val.append({'Acc':acc, 'MAPE':mape, 'Parameters':p}, ignore_index=True)

        # model fitting
        m = Prophet(interval_width=0.95)
        m.fit(df)
        future = m.make_future_dataframe(periods=72, freq='1h')
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


def create_write_influxdb_data(forecast, tagKey, tagValue, interval):
    ds = forecast.ds.values

    # yhat
    values = forecast.yhat.values
    data = {
        'ds': ds,
        'count': values,
        'variable': ['yhat'] * len(values),
        tagKey: [tagValue] * len(values),
        'interval': [interval] * len(values)
    }
    yhat_df = pd.DataFrame(data)
    yhat_df.set_index('ds', inplace=True)

    # yhat_lower
    values = forecast.yhat_lower.values
    data = {
        'ds': ds,
        'count': values,
        'variable': ['yhat_lower'] * len(values),
        tagKey: [tagValue] * len(values),
        'interval': [interval] * len(values)
    }
    yhat_lower_df = pd.DataFrame(data)
    yhat_lower_df.set_index('ds', inplace=True)

    # yhat_upper
    values = forecast.yhat_upper.values
    data = {
        'ds': ds,
        'count': values,
        'variable': ['yhat_upper'] * len(values),
        tagKey: [tagValue] * len(values),
        'interval': [interval] * len(values)
    }
    yhat_upper_df = pd.DataFrame(data)
    yhat_upper_df.set_index('ds', inplace=True)

    return {
        'yhat': yhat_df,
        'yhat_lower': yhat_lower_df,
        'yhat_upper': yhat_upper_df
    }


def write_influxdb(data, measurement, tagKey):
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)

        yhat_df = data['yhat']
        print('yhat_df', yhat_df)

        yhat_lower_df = data['yhat_lower']
        print('yhat_lower_df', yhat_lower_df)

        yhat_upper_df = data['yhat_upper']
        print('yhat_upper_df', yhat_upper_df)

        write_api.write(bucket=BUCKET, org=ORG, record=yhat_df,
                        data_frame_measurement_name=measurement+'-prophet-forecast', data_frame_tag_columns=['variable', tagKey, 'interval'])
        write_api.write(bucket=BUCKET, org=ORG, record=yhat_lower_df,
                        data_frame_measurement_name=measurement+'-prophet-forecast', data_frame_tag_columns=['variable', tagKey, 'interval'])
        write_api.write(bucket=BUCKET, org=ORG, record=yhat_upper_df,
                        data_frame_measurement_name=measurement+'-prophet-forecast', data_frame_tag_columns=['variable', tagKey, 'interval'])

        write_api.close()

    except Exception as e:
        print('write_influxdb error:', str(e))


def run(event, context):
    print('event:', event)
    print('context:', context)

    current_time = datetime.datetime.now().time()
    name = context.function_name

    body = {
        'message': 'Your cron function ' + name + ' ran at ' + str(current_time),
        'input': event
    }

    # delete future data before forecast again
    delete_data()

    query = create_query('ras', 'pid', 'ITMCHECKM', '1h')

    # load the data
    query_influxdb_res = query_influxdb(query)
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

    data = create_write_influxdb_data(forecast, 'pid', 'ITMCHECKM', '1h')

    write_influxdb(data, 'ras', 'pid')

    response = {
        'statusCode': 200,
        'body': json.dumps(body)
    }

    client.close()

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
