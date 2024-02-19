import datetime

def get_time_stamp():
    return '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())


