import whisper
import tempfile
import os
import ffmpeg
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login
import torch

from huggingface_hub import login
#hf_token=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
login(hf_token)
processor=WhisperProcessor.from_pretrained("Laith05/roma-whisper-small-levantine")
model=WhisperForConditionalGeneration.from_pretrained("Laith05/roma-whisper-small-levantine")
model.eval()

F1_PROMPT = """
فورمولا 1، سباق، سائق، فريق، بطولة، تصفيات، حلبة،
فيرستابن، هاميلتون، لوكليرك، نوريس، بيريز، ألونسو،
ريد بول، فيراري، مرسيدس، ماكلارين، أستون مارتن،
موناكو، سيلفرستون، مونزا، سوزوكا، أبوظبي، جدة
"""

def convert_audio(input_path,output_path):
    (
        ffmpeg
        .input(input_path)
        .output(output_path, ar=16000, ac=1, format='wav')
        .run(quiet=True, overwrite_output=True)
    )

def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    try:
        convert_audio(tmp_in_path, tmp_out_path)

        import librosa
        audio, _ = librosa.load(tmp_out_path, sr=16000)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="arabic",
            task="transcribe"
        )

        # Encode F1 vocabulary prompt correctly for HuggingFace
        prompt_ids = processor.get_prompt_ids(F1_PROMPT, return_tensors="pt")

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                prompt_ids=prompt_ids,      # ← fixed
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()

    except Exception as e:
        print(f"ASR error: {e}")
        return ""

    finally:
        os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path): 
            os.unlink(tmp_out_path)
