import asyncio
import websockets
import sounddevice as sd
import numpy as np

async def send_audio():
    """
    Async function to send audio data to the WebSocket server and receive prediction results.

    Raises:
        websockets.exceptions.ConnectionClosedError: Raised when the WebSocket connection is closed by the server.
        Exception: Raised for any other exceptions.
        KeyboardInterrupt: Raised when the user interrupts the program.
    """
    async with websockets.connect('ws://localhost:8000/audio') as websocket:
        print("Connected to WebSocket server")

        try:
            while True:
                duration = 3  # Duration of audio capture in seconds
                sample_rate = 44100  # Sample rate of the audio

                # Start capturing audio from the microphone
                print("Recording audio...")
                audio_data = sd.rec(duration * sample_rate, samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()

                # Convert audio data to bytes
                audio_bytes = audio_data.tobytes()

                # Send audio data to the WebSocket server
                await websocket.send(audio_bytes)
                print("Audio data sent")

                # Receive the prediction result from the server
                prediction = await websocket.recv()
                print(f"Received prediction: {prediction}")

        except websockets.exceptions.ConnectionClosedError:
            print("WebSocket connection closed by the server")

        except Exception as e:
            print(f"Error occurred: {str(e)}")

        except KeyboardInterrupt:
            # Close the WebSocket connection
            await websocket.close()

asyncio.get_event_loop().run_until_complete(send_audio())
