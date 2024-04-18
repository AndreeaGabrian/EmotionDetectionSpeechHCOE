from transformers import pipeline
from emotion_polarity_detection import run_emotion_model
from utils import json2excel

# models for emotion
# model_pipe = pipeline("text-classification", model="nickwong64/bert-base-uncased-poems-sentiment")
# run_emotion_model(model_pipe, "transcription_results/transcription_p1_conv_results_whisper_BASE.json", "emotion_results/emotion_results_p1_conv.json")
# run_emotion_model(model_pipe, "transcription_results/transcription_p2_conv_results_whisper_BASE.json", "emotion_results/emotion_results_p2_conv.json")

json2excel("emotion_results/emotion_results_p1_conv")
json2excel("emotion_results/emotion_results_p2_conv")