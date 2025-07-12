from flask import Flask, request, send_file
import os
import uuid
import torch
from fairseq.data.dictionary import Dictionary
from rvc_python.infer import RVCInference
from types import MethodType
from scipy.io import wavfile

# Allow fairseq dictionary during safe deserialization
torch.serialization.add_safe_globals([Dictionary])

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "cloned_audio_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Patch method to use local model
def patched_infer_file(self, input_path, output_path):
    if not self.current_model:
        raise ValueError("Model not loaded.")
    model_info = self.models[self.current_model]
    file_index = model_info.get("index", "")

    result = self.vc.vc_single(
        sid=0,
        input_audio_path=input_path,
        f0_up_key=self.f0up_key,
        f0_method=self.f0method,
        file_index=file_index,
        index_rate=self.index_rate,
        filter_radius=self.filter_radius,
        resample_sr=self.resample_sr,
        rms_mix_rate=self.rms_mix_rate,
        protect=self.protect,
        f0_file="",
        file_index2=""
    )

    wav = result[0] if isinstance(result, tuple) else result
    wavfile.write(output_path, self.vc.tgt_sr, wav)

@app.route("/")
def health_check():
    return {"status": "Model API is running"}

@app.route("/clone", methods=["POST"])
def clone_voice():
    audio = request.files.get("audio")
    model = request.files.get("model")

    if not audio or not model:
        return {"error": "Missing audio or model file"}, 400

    # Save inputs
    input_audio_path = os.path.join(UPLOAD_DIR, f"input_{uuid.uuid4().hex}.wav")
    model_path = os.path.join(UPLOAD_DIR, f"model_{uuid.uuid4().hex}.pth")
    output_path = os.path.join(UPLOAD_DIR, f"output_{uuid.uuid4().hex}.wav")

    audio.save(input_audio_path)
    model.save(model_path)

    # Run inference
    rvc = RVCInference(model_path=model_path)
    rvc.infer_file = MethodType(patched_infer_file, rvc)
    rvc.set_params(f0up_key=0, index_rate=0.75)
    rvc.infer_file(input_path=input_audio_path, output_path=output_path)

    return send_file(output_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
