import os

import query1
import query2


QUERY1_ENDPOINT = '/data/1/'
QUERY2_ENDPOINT = '/data/2/'

if __name__ == "__main__":
    host = os.getenv('BENCHMARK_SYSTEM_URL')
    if not host:
        host = 'localhost'
        print('Warning: Benchmark system url undefined. Using localhost!')
    if host is None or '':
        print('Error reading Server address!')
    query1.run(host, QUERY1_ENDPOINT)
    query2.run(host, QUERY2_ENDPOINT)
    print('Solution Done')