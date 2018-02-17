from sys import stdout as so
import time


class Progresser:
    def __init__(self, total_num, msg=''):
        self.total = total_num
        self.num = 0
        self.start_time = time.time()
        self.msg = msg

    def count(self):
        self.show_progress(self.num)
        self.num += 1

    def show_progress(self, current_num):
        if current_num % 10 == 0:
            eltime = time.time() - self.start_time
            retime = (self.total - current_num - 1) * eltime / (current_num + 1)

            el_str = str(int(eltime / 3600)) + ':' + str(int((eltime % 3600) / 60)) + ':' + str(int(eltime % 60))
            re_str = str(int(retime / 3600)) + ':' + str(int((retime % 3600) / 60)) + ':' + str(int(retime % 60))

            so.write('\r' + self.msg + '\ttime: ' + el_str + ' + ' + re_str
                     + '\t\tprogress: %' + str(round(100 * (current_num + 1) / self.total, 2)) + '\t')
