from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["happy funk with saxophone"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)


sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out4.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())