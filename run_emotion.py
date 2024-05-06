from transformers import pipeline
from emotion_polarity_detection import run_emotion_model
from utils import json2excel

# models for emotion
model_pipe = pipeline("text-classification", model="nickwong64/bert-base-uncased-poems-sentiment")
model_name = "nickwong64/bert-base-uncased-poems-sentiment"
run_emotion_model(model_pipe, "transcription_results/transcription_p1_conv_results_whisper_BASE_new.json", "emotion_results/emotion_results_p1_conv_new.json", model_name)
run_emotion_model(model_pipe, "transcription_results/transcription_p2_conv_results_whisper_BASE_new.json", "emotion_results/emotion_results_p2_conv_new.json", model_name)

json2excel("emotion_results/emotion_results_p1_conv_new")
json2excel("emotion_results/emotion_results_p2_conv_new")