import time

time.time()

def cur_date():

    day = 24
    hour = 60
    minute = 60
    current = time.time()
    t_sec = current % day*hour*minute
    cur_hours = int(t_sec / (hour*minute))
    t_minutes = t_sec / 60
    cur_mins = int(t_minutes % 60)
    cur_sec = int(t_sec % 60)

    days = int(current / 86400)

    print('Time', cur_hours, ':', cur_mins, ':', cur_sec, 'Days', days)




