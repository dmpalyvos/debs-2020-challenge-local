import requests
import socket
import time

# Timeout for GET requests to the grader.
# Especially important for the first request, when the containers are still starting
GET_TIMEOUT = 600
# Maximum wait time for grader to become available
MAX_WAIT_BENCHMARK_SECONDS = 1800

def host_url(host, endpoint):
    return "http://" + host + endpoint


def get_batch(host, endpoint):
    return requests.get(host_url(host, endpoint), timeout=GET_TIMEOUT)


def post_result(host, endpoint, payload):
    headers = {'Content-type': 'application/json'}
    return requests.post(host_url(host, endpoint), json=payload, headers=headers)


def waitForBenchmarkSystem(host, port):
    print(f'Waiting for {host}:{port} to become available...')
    s = socket.socket()
    startTime = time.time()
    while time.time() < startTime + MAX_WAIT_BENCHMARK_SECONDS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex((host, port))
            if result == 0:
                print(f'Connected to {host}:{port}')
                return
            else:
                time.sleep(10)
    print(f'Failed to establish connection to {host}:{port} after {MAX_WAIT_BENCHMARK_SECONDS}')