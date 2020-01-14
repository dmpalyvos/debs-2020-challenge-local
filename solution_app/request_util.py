import requests

# Timeout for GET requests to the grader.
# Especially important for the first request, when the containers are still starting
GET_TIMEOUT = 600

def host_url(host, endpoint):
    return "http://" + host + endpoint


def get_batch(host, endpoint):
    return requests.get(host_url(host, endpoint), timeout=GET_TIMEOUT)


def post_result(host, endpoint, payload):
    headers = {'Content-type': 'application/json'}
    return requests.post(host_url(host, endpoint), json=payload, headers=headers)