# DEBS 2020 Grand Challenge HTTP-Client Example Kit

This repository contains an example HTTP-client that connects you to the DEBS 2020 Grand Challenge Benchmark System.

Please use this repository as a template for your work. The final Benchmark System will be mostly the same to the one in this repository.

We use [Docker Compose](https://docs.docker.com/compose/) to reduce the complexity of integration with the Benchmark System.
Please read the instructions below to get an insight about how you can get started.

## About this repository

This repository contains the basic project structure for your implementation.

- `dataset` is the folder that should contain the provided training datasets `in1.csv` (Query 1), `in2.csv` (Query 2) and `out.csv`. 
- `solution_app` is the folder of the solution implementation files.
- `docker-compose.yml` - defines the services that run together (HTTP-client against our Benchmarking system).
- `Dockerfile.solution` - defines the steps needed to build the container with your solution. *If you decided to use another language than Python, you will need to redefine this file appropriately*.

## Before you start

Make sure you have Docker Engine and Docker Compose installed. You may use the official links below for downloading:

[Download Docker Engine](https://docs.docker.com/get-started/#prepare-your-docker-environment)

[Download Docker Compose](https://docs.docker.com/compose/install/#install-compose)

Check your installation:

```bash
  docker --version
  docker-compose --version
```

## How to get started

You need to implement your solution as an HTTP-client. Sample solutions for Query 1 and Query 2, written in Python, are already provided in the `/solution_app` folder. However, you are free to use the language and framework of your preference.

1. Clone this repository.
1. Use the project structure provided. Place `in1.csv`, `in2.csv` and `out.csv` in the `/dataset` folder so that the Benchmark System Container can evaluate your solution.
1. Implement your HTTP-client as REST web service, that reaches via GET and POST requests (you may see an example implementation in `solution.py`).

    - This means that your solution should request data via a GET method, and submit your answer via a POST method.

    - Use the `/data/1/` path to send and receive data for Query 1 and `/data/2/` for Query 2.

    - For each GET request you will receive a new chunk of data containing various number of tuples.

    - You need to submit your answer for this chunk via POST request.

    - After getting all chunks, your solution should stop upon seeing `404` status code. Now you ready for the next step.

1. Both the final, and Benchmark Systems you will test against, in their environments, will contain `BENCHMARK_SYSTEM_URL` as an environment variable, so make sure you read it in your solution program to be able to connect to to our system.
1. Define the procedure to install the dependencies, compile and run your solution in `Dockerfile.solution`. (If you use python, you will might only need to update the `requirements.txt` in the `solution_app` directory.)
1. To start the evaluation of your solution run:

      ```bash
      docker-compose up
      ```

1. Check the logs of `'benchmark-grader'` Docker container to see details of your run.
    Use this command:

      ```bash
      docker logs benchmark-system
      ```

1. Make changes to your system if needed.
After any change to your prediction system or HTTP-client, please run these commands:

      ```bash
      docker-compose down
      ```

    This will stop the previous run of benchmark system. Then run:

      ```bash
      docker-compose up --build
      ```

    To rebuild with changes you made.

`Note`: As mentioned above, if you want to use another language instead of Python, you need to change the content of `Dockerfile.solution` to support language of your choice.


### Grader REST Paths

- Data path for Query 1: `BENCHMARK_SYSTEM_URL/data/1/`
- Data path for Query 2: `BENCHMARK_SYSTEM_URL/data/2/`
- Data path for retrieving full score: `BENCHMARK_SYSTEM_URL/score/all`

## Standalone testing

If you want to test your solution outside docker (e.g., to speed up development in the initial stages) you can do so as follows.

Start the grader container and forward port 80:

```bash
docker run -p 80:80 palyvos/debs-2020-challenge-grader
```

After that, your solution, running locally, should be able to access the grader exactly as it does when running in a container. Note that you will need to restart the grader container between consequent invocations of your solution application.

