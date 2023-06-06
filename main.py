from fastapi import FastAPI, WebSocketDisconnect
from fastapi import WebSocket
import numpy as np
import matplotlib.pyplot as plt

from ML_model import process_audio_data

app = FastAPI()

# WebSocket endpoint to receive and process audio data
@app.websocket("/audio")
async def audio(websocket: WebSocket):
    """
    WebSocket endpoint to receive and process audio data.

    Args:
        websocket (WebSocket): WebSocket connection object.

    Raises:
        WebSocketDisconnect: Raised when the WebSocket connection is disconnected.
    """
    try:
        await websocket.accept()

        while True:
            sample_rate = 44100  # Sample rate of the audio
            audio_bytes = await websocket.receive_bytes()
            print("Audio data received")

            # ---- For plotting ---- |
            # ---------------------- v

            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

            # Generate the time axis based on the sample rate and number of samples
            duration = len(audio_data) / sample_rate
            time = np.linspace(0.0, duration, len(audio_data))

            # Plot the waveform
            plt.plot(time, audio_data)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Audio Waveform')
            plt.show()

            # ---------------------- ^
            # ---- For plotting ---- |

            # Process the audio data with your ML model
            prediction = process_audio_data(audio_bytes, sample_rate)
            print(f"Sending a prediction: {prediction}")

            # Send the prediction result back to the WebSocket client
            await websocket.send_text(prediction)
    except WebSocketDisconnect as e:
        # Handle the WebSocket disconnection gracefully
        print(f"WebSocket disconnected with code: {e.code}")
