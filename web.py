
import argparse
import os
import whisper
import torch

from datetime import datetime
from flask import Flask, request, abort
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'console': {
        'class': 'logging.StreamHandler',
        'stream'  : 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['console']
    }
})
app = Flask(__name__)

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true', help="Don't use the english model.")
    args = parser.parse_args()

    # Load model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"

    global audio_model
    audio_model = whisper.load_model(model)
    app.logger.info("Model loaded: %s", model)

@app.route('/')                
def transcribe():
    wav_file_path = request.args.get('wav_file_path')
    app.logger.info('wav_file_path: %s', wav_file_path)
    timestamp = datetime.utcnow()

    if not wav_file_path: 
        app.logger.error('Not found: %s', wav_file_path)
        abort(400)

    if not os.path.exists(wav_file_path):
        app.logger.error('Not exist: %s', wav_file_path)
        abort(404)
    
    result = audio_model.transcribe(wav_file_path, fp16=torch.cuda.is_available())
    s = result['text'].strip()
    app.logger.info('[%f] %s', (datetime.utcnow() - timestamp).total_seconds(), s)
    return s

if __name__ == '__main__':
    init()
    app.run()