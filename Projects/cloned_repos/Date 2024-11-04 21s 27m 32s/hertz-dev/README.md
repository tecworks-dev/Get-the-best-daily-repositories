# hertz-dev

Hertz-dev is an open-source, first-of-its-kind base model for full-duplex conversational audio.

See our blog post for more details: https://si.inc/hertz-dev/

We recommend starting by using `inference.ipynb` to generate one- or two-channel completions from a prompt.

Then, you can use `inference_client.py` and `inference_server.py` to talk to the model live through your microphone.
These are currently experimental, and have primarily been tested with Ubuntu on the server and MacOS on the client.

All three scripts will automatically download the models to the `./ckpt` directory.