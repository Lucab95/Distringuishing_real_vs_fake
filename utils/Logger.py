# class Logger(object):

#     def __init__(self, path, header):
#         self.log_file = open(path, 'w')
#         self.logger = csv.writer(self.log_file, delimiter='\t')

#         self.logger.writerow(header)
#         self.header = header

#     def __del(self):
#         self.log_file.close()

#     def log(self, values):
#         write_values = []
#         for col in self.header:
#             assert col in values
#             write_values.append(values[col])

#         self.logger.writerow(write_values)
#         self.log_file.flush()
import sys
class Logger():
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass