import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=10)  # generate amount of seconds.

descriptions = ['upbeat happy soul']

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, progress=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write("output/bach", one_wav.cpu(), model.sample_rate, strategy="loudness")