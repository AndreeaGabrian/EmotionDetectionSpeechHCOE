from transformers import AutoProcessor, WhisperForConditionalGeneration
from process_trimm_files import process_trim_files

# model for transcription
processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")

input_folder_path = "p2_conv"
output_folder_path = "transcription_results"

process_trim_files(input_folder_path, processor, model, output_folder_path, "p2_conv")


