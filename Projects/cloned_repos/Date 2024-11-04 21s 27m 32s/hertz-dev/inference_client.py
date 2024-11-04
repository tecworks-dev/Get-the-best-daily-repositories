# server.py remains the same as before

# Updated client.py
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import base64
import queue
import argparse
import requests
import time

class AudioClient:
    def __init__(self, server_url="ws://localhost:8000", token_temp=None, categorical_temp=None, gaussian_temp=None):
        # Convert ws:// to http:// for the base URL
        self.base_url = server_url.replace("ws://", "http://")
        self.server_url = f"{server_url}/audio"
        
        # Set temperatures if provided
        if any(t is not None for t in [token_temp, categorical_temp, gaussian_temp]):
            self.set_temperature_and_echo(token_temp, categorical_temp, gaussian_temp)
        
        # Initialize queues
        self.audio_queue = queue.Queue()
        self.output_queue = queue.Queue()
    
    def set_temperature_and_echo(self, token_temp=None, categorical_temp=None, gaussian_temp=None, echo_testing = False):
        """Send temperature settings to server"""
        params = {}
        if token_temp is not None:
            params['token_temp'] = token_temp
        if categorical_temp is not None:
            params['categorical_temp'] = categorical_temp
        if gaussian_temp is not None:
            params['gaussian_temp'] = gaussian_temp
            
        response = requests.post(f"{self.base_url}/set_temperature", params=params)
        print(response.json()['message'])
    
    def audio_callback(self, indata, frames, time, status):
        """This is called for each audio block"""
        if status:
            print(status)
        # if np.isclose(indata, 0).all():
        #     raise Exception('Audio input is not working - received all zeros')
        # Convert float32 to int16 for efficient transmission
        indata_int16 = (indata.copy() * 32767).astype(np.int16) 
        # indata_int16 = np.zeros_like(indata_int16)
        self.audio_queue.put(indata_int16)
    
    def output_stream_callback(self, outdata, frames, time, status):
        """Callback for output stream to get audio data"""
        if status:
            print(status)
        
        try:
            data = self.output_queue.get_nowait()
            data = data.astype(np.float32) / 32767.0
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:len(outdata)]
        except queue.Empty:
            outdata.fill(0)
    
    async def process_audio(self):
        async with websockets.connect(self.server_url) as ws:
            while self.running:
                if not self.audio_queue.empty():
                    # Get recorded audio
                    audio_data = self.audio_queue.get()
                    print(f'Data from microphone:{audio_data.shape, audio_data.dtype, audio_data.min(), audio_data.max()}')
                    
                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                    
                    # Send to server
                    time_sent = time.time()
                    await ws.send(f"data:audio/raw;base64,{audio_b64}")
                    
                    # Receive processed audio
                    response = await ws.recv()
                    response = response.split(",")[1]
                    time_received = time.time()
                    print(f"Data sent: {audio_b64[:10]}. Data received: {response[:10]}. Received in {(time_received - time_sent) * 1000:.2f} ms")
                    processed_audio = np.frombuffer(
                        base64.b64decode(response),
                        dtype=np.int16
                    ).reshape(-1, CHANNELS)
                    print(f'Data from model:{processed_audio.shape, processed_audio.dtype, processed_audio.min(), processed_audio.max()}')
                    
                    self.output_queue.put(processed_audio)
    
    def start(self):
        self.running = True
        # Print audio device information
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        
        print("\nAudio Device Configuration:")
        print("-" * 50)
        print(f"Default Input Device:\n{default_input}\n")
        print(f"Default Output Device:\n{default_output}\n") 
        print("\nAll Available Devices:")
        print("-" * 50)
        for i, device in enumerate(devices):
            print(f"Device {i}:")
            print(f"Name: {device['name']}")
            print(f"Channels (in/out): {device['max_input_channels']}/{device['max_output_channels']}")
            print(f"Sample Rates: {device['default_samplerate']}")
            print()
        input_device = input("Enter the index of the input device or press enter for default: ")
        output_device = input("Enter the index of the output device or press enter for default: ")
        if input_device == "":
            input_device = default_input['index']
        if output_device == "":
            output_device = default_output['index']
        with sd.InputStream(callback=self.audio_callback,
                          channels=CHANNELS,
                          samplerate=SAMPLE_RATE,
                          device=int(input_device),
                          blocksize=2000), \
             sd.OutputStream(callback=self.output_stream_callback,
                           channels=CHANNELS,
                           samplerate=SAMPLE_RATE,
                           blocksize=2000,
                           device=int(output_device)):
            
            asyncio.run(self.process_audio())
    
    def stop(self):
        self.running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Client with Temperature Control')
    parser.add_argument('--token_temp', '-t1', type=float, help='Token (LM) temperature parameter')
    parser.add_argument('--categorical_temp', '-t2', type=float, help='Categorical (VAE) temperature parameter')
    parser.add_argument('--gaussian_temp', '-t3', type=float, help='Gaussian (VAE) temperature parameter')
    parser.add_argument('--server', '-s', default="ws://localhost:8000", 
                        help='Server URL (default: ws://localhost:8000)')
    
    args = parser.parse_args()
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    client = AudioClient(
        server_url=args.server,
        token_temp=args.token_temp,
        categorical_temp=args.categorical_temp,
        gaussian_temp=args.gaussian_temp
    )
    
    try:
        client.start()
    except KeyboardInterrupt:
        client.stop()