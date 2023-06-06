# Audio Classification WebSocket Server

This project implements a WebSocket server that receives audio data, performs audio classification using a dummy machine learning model, and sends back the classification result to the client. The server is built using FastAPI and supports real-time audio streaming through WebSocket connections.

## Features

- Accepts audio data via WebSocket connection
- Processes the received audio data using a dummy machine learning model
- Returns the classification result back to the client
- Supports real-time audio streaming
- Includes a client script for testing the server

## Prerequisites

Before running the server, make sure you have the following dependencies installed:

- Python 3.7 or above
- FastAPI
- websockets
- numpy
- matplotlib
- librosa (for audio feature extraction)
- scikit-learn (for the dummy machine learning model)
- sounddevice (for audio recording in the client script)

You can install the dependencies using pip:
```
pip install fastapi websockets numpy matplotlib librosa scikit-learn sounddevice
```
#### Using pipenv

If you prefer to use `pipenv` for managing virtual environments, follow these steps:

1. Install `pipenv` using `pip`:
```
pip install pipenv
```


2. Navigate to the project directory and run the following command to create a virtual environment and install the dependencies:
```
pipenv install --dev
```
This will create a virtual environment and install both the project dependencies and development dependencies specified in the `Pipfile`.

To activate the virtual environment, you can use the following command:
```
pipenv shell
```


Make sure to activate the virtual environment before running the server or the client script.

## Usage

1. Clone this repository to your local machine.

2. Navigate to the project directory:
```
cd audio-classification-websocket
```

3. Start the WebSocket server by running the following command:
```
uvicorn main:app --reload
```
The server will start running at `http://localhost:8000`.

4. Open a web browser or use a WebSocket client to connect to `ws://localhost:8000/audio`.

5. Use the client script `test.py` to send audio data to the server for classification. Run the script using the following command:
```
python test.py
```

The script will start recording audio from the microphone, send it to the server, and display the classification result received from the server.

## Directory Structure

The project directory contains the following files:

- `ML_model.py`: Contains the dummy machine learning model and audio processing functions used by the server.

- `main.py`: Implements the WebSocket server using FastAPI and defines the endpoint for audio data.

- `test.py`: Client script to record audio and send it to the server for classification.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

