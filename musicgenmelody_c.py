import os
import ssl
import certifi
import gradio as gr
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from transformers import pipeline

# Fix SSL issues
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Load models
model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=10)
story_generator = pipeline("text-generation", model="gpt2")
summary_generator = pipeline("summarization", model="facebook/bart-large-cnn")
emotion_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Emotion detection from description
def predict_emotion_from_description(description):
    labels = ["happy", "sad", "angry", "relaxed", "epic", "calm", "hopeful", "melancholic", "energetic"]
    result = emotion_classifier(description, labels)
    return f"Top emotion: {result['labels'][0]} ({result['scores'][0]:.2f})"

# Generation function

def generate_music_and_story(description, melody_file):
    if not melody_file:
        return None, "No file uploaded", "", "", None, None

    try:
        # Load melody
        melody, sr = torchaudio.load(melody_file)

        # Generate music
        wav = model.generate_with_chroma([description], melody[None], sr)
        audio_path = "output/music"
        audio_write(audio_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
        audio_file_path = audio_path + ".wav"

        # Generate full story
        prompt = f"Write a short story inspired by this music description: {description}. Let the story have a beginning, middle, and end."
        story = story_generator(prompt, max_length=300, do_sample=True)[0]["generated_text"]

        # Save full story
        story_path = "output/story.txt"
        with open(story_path, "w") as f:
            f.write(story)

        # Summarize story for display
        summary = summary_generator(story, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

        # Predict emotion from description
        emotion_summary = predict_emotion_from_description(description)

        return audio_file_path, summary, emotion_summary, "", audio_file_path, story_path

    except Exception as e:
        return None, f"Error: {str(e)}", "", "", None, None


# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# T.A.C.O\n A music composer copyright-free that also creates stories and defines their mood ")

    with gr.Row():
        description = gr.Textbox(label="description (input):", placeholder="e.g. a child that was trapped in a rock festival")
        melody_file = gr.Audio(label="upload your base melody:", type="filepath")

    generate_btn = gr.Button("Create!")

    output_audio = gr.Audio(label="generated song:")
    output_story = gr.Textbox(label="generated story:", lines=6)
    output_emotion = gr.Textbox(label="Mood:", interactive=False)

    with gr.Row():
        audio_download = gr.File(label="download here your song")
        story_download = gr.File(label="download here your story")

    generate_btn.click(
        fn=generate_music_and_story,
        inputs=[description, melody_file],
        outputs=[output_audio, output_story, output_emotion, audio_download, story_download]
    )
demo.launch(share=True)
