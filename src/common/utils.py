def calc_abspos(timestamp, origin_point):
    try:
        return (timestamp - origin_point).total_seconds() / 60 / 60
    except:  # Sometimes have to convert to datetime
        return (timestamp - origin_point).dt.total_seconds() / 60 / 60
