try:
  import unzip_requirements
except ImportError:
  pass

import boto3
import datetime
import logging
import pandas as pd
# from prophet import Prophet

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


def run(event, context):
    current_time = datetime.datetime.now().time()
    name = context.function_name
    logger.info("Your cron function " + name + " ran at " + str(current_time))

    timestamps = get_metric_data()['MetricDataResults'][0]['Timestamps']
    values = get_metric_data()['MetricDataResults'][0]['Values']

    data = {
        'timestamps': timestamps,
        'values': values
    }
    df = pd.DataFrame(data)
    print(df)
