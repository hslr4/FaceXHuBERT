import librosa
import argparse
import pickle
import torch
import numpy as np
import os
from faceXhubert import FaceXHuBERT
from transformers import Wav2Vec2Processor

import uvicorn
from fastapi import FastAPI, Response, Request


parser = argparse.ArgumentParser(description='FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis using Self-Supervised Speech Representation Learning')
parser.add_argument("--model_name", type=str, default="FaceXHuBERT")
parser.add_argument("--dataset", type=str, default="BIWI", help='name of the dataset folder. eg: BIWI')
parser.add_argument("--fps", type=float, default=25, help='frame rate - 25 for BIWI')
parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex Decoder hidden size')
parser.add_argument("--vertice_dim", type=int, default=70110, help='number of vertices - 23370*3 for BIWI')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--train_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
parser.add_argument("--test_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
parser.add_argument("--wav_path", type=str, default="demo/wav/server.wav", help='path of the input audio signal in .wav format')
parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions in .npy format')
parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
parser.add_argument("--template_path", type=str, default="templates_scaled.pkl", help='path of the personalized templates')
parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI topology')
parser.add_argument("--input_fps", type=int, default=50, help='HuBERT last hidden state produces 50 fps audio representation')
parser.add_argument("--output_fps", type=int, default=25, help='fps of the visual data, BIWI was captured in 25 fps')
parser.add_argument("--emotion", type=int, default="1", help='style control for emotion, 1 for expressive animation, 0 for neutral animation')

parser.add_argument("--host", type=str, default='0.0.0.0', help='host to expose rest api')
parser.add_argument("--port", type=int, default=7222, help='port to expose rest api')

args = parser.parse_args()

WAV_PATH = args.wav_path

template_file = os.path.join(args.dataset, args.template_path)
with open(template_file, 'rb') as fin:
    templates = pickle.load(fin,encoding='latin1')

train_subjects_list = [i for i in args.train_subjects.split(" ")]

one_hot_labels = np.eye(len(train_subjects_list))
emo_one_hot_labels = np.eye(2)

with torch.no_grad():
    model = FaceXHuBERT(args)
    model.load_state_dict(torch.load('pretrained_model/{}.pth'.format(args.model_name)))
    model = model.to(torch.device(args.device))
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Lipsync"}


@app.post("/lipsync/", response_class=Response)
async def lipsync(data: Request, emotion: int = args.emotion):
    data_b = await data.body()
    with open(WAV_PATH, mode='bw') as f:
        f.write(data_b)

    with torch.no_grad():
        if emotion == 1:
            emo_one_hot = torch.FloatTensor(emo_one_hot_labels[1]).to(device=args.device)
        else:
            emo_one_hot = torch.FloatTensor(emo_one_hot_labels[0]).to(device=args.device)
        
        iter = train_subjects_list.index(args.condition)
        one_hot = one_hot_labels[iter]
        one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
        one_hot = torch.FloatTensor(one_hot).to(device=args.device)

        temp = templates[args.subject]
                
        template = temp.reshape((-1))
        template = np.reshape(template,(-1,template.shape[0]))
        template = torch.FloatTensor(template).to(device=args.device)

        speech_array, sampling_rate = librosa.load(os.path.join(WAV_PATH), sr=16000)
        audio_feature = processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values
        audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

        prediction = model.predict(audio_feature, template, one_hot, emo_one_hot)
        prediction = prediction.squeeze()
        prediction = prediction.detach().cpu().numpy()
        print('prediction.shape', prediction.shape)
        torch.cuda.empty_cache()
        return Response(content=prediction.tobytes(), media_type='application/octet-stream')
    

if __name__ == '__main__':
    uvicorn.run(app, port=args.port, host=args.host)