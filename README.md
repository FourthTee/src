These are the instructions to deploy the tensor-serving using docker

1. Direct to the directory containing the files

2. Clone this repo using the command 

```
$ git clone https://github.com/FourthTee/src.git

```
3. Convert the frozen graph using the python script

```
$ python3 src/convert_pb.py

```

3. Pull the docker image from the docker hub

```
$ docker pull fourthtee/tf-serving-sertis:sub1

```
4. Run the docker to start the API

```
$ docker run -p 8501:8501 --mount type=bind,source=/home/fourth/Desktop/sertis-mle/saved/,target=/models/test_model/1 -e MODEL_NAME=test_model -t fourthtee/tf-serving-sertis:sub1

```
5. To use the tensorflow serving use the script request.py


```
$ python3 src/request.py
```
