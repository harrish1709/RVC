from flask import Flask, request, send_file, jsonify
import os, uuid, torch, gdown
from fairseq.data.dictionary import Dictionary
from rvc_python.infer import RVCInference
from types import MethodType
from scipy.io import wavfile

torch.serialization.add_safe_globals([Dictionary])

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "cloned_audio_files")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Map model name to Google Drive file ID
MODEL_SOURCES = {
    "speaker1": "1mRbQIQ5owj8e_0SQxnXGLRhqlVKBLm5Y",  # Replace with real file IDs
    "speaker2": "1sYP_wb1-zxew2eIQPq29gPBlu8RFvdDz",
    "speaker3": "1rcTRcrKpAgyX7fyRgd3CkaQ2yfa9ckgj",
    "speaker4": "1YJWhFY6wqJVF1N3U7CN4RhZSQBIrZK9j",
    "speaker5": "10QhF6S5rRzBDYjxCEa_F_oWF3rfQ0Ke6",
}

# Download model from Google Drive if not present
def download_model_if_missing(model_name, gdrive_id):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"Downloading model: {model_name}")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

# Patch inference method
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
    return jsonify({"status": "Model API running", "available_models": list(MODEL_SOURCES.keys())})

@app.route("/clone", methods=["POST"])
def clone_voice():
    audio = request.files.get("audio")
    model_name = request.form.get("model_name")

    if not audio or not model_name:
        return {"error": "Missing audio or model_name"}, 400

    if model_name not in MODEL_SOURCES:
        return {"error": f"Model '{model_name}' not found"}, 404

    input_audio_path = os.path.join(UPLOAD_DIR, f"input_{uuid.uuid4().hex}.wav")
    output_path = os.path.join(UPLOAD_DIR, f"output_{uuid.uuid4().hex}.wav")
    audio.save(input_audio_path)

    try:
        model_path = download_model_if_missing(model_name, MODEL_SOURCES[model_name])
        rvc = RVCInference(model_path=model_path)
        rvc.infer_file = MethodType(patched_infer_file, rvc)
        rvc.set_params(f0up_key=0, index_rate=0.75)
        rvc.infer_file(input_path=input_audio_path, output_path=output_path)
    except Exception as e:
        return {"error": str(e)}, 500

    return send_file(output_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
