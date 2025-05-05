import ssl
import certifi
import os

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Optional: text generation
from transformers import pipeline

# Set up text generation pipeline (can switch to GPT-Neo or other models)
story_generator = pipeline("text-generation", model="gpt2")

# Initialize MusicGen
model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=10)

descriptions = ['sad rock']
melody, sr = torchaudio.load('./assets/bach.mp3')

# Generate music
wav = model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, progress=True)

# Make sure output directory exists
os.makedirs("output", exist_ok=True)

for idx, one_wav in enumerate(wav):
    audio_path = f"output/music_{idx}.wav"
    audio_write(audio_path, one_wav.cpu(), model.sample_rate, strategy="loudness")

    # Generate a story based on the description
    story_prompt = f"Write a vivid short story based on the theme: {descriptions[idx]}"
    story = story_generator(story_prompt, max_length=150, do_sample=True)[0]["generated_text"]

    # Save story to a .txt file
    story_path = f"output/music_{idx}_story.txt"
    with open(story_path, "w") as f:
        f.write(story)

    print(f"Saved: {audio_path}")
    print(f"Saved: {story_path}")
