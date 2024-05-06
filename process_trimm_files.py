import json
import re
import time
import os
import torchaudio
from torchaudio.transforms import Resample
from utils import json2excel


# sorts alphanumeric the files in data = list of files name
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def process_trim_files(input_folder_path, processor, model, output_folder_path, audio_name, model_name):
    # Get a sorted list of files in the folder
    files = os.listdir(input_folder_path)
    file_names = sorted_alphanumeric(files)
    # Create a dictionary to store transcriptions
    transcriptions_dict = {}
    exec_times_list = []
    # Consolidate consecutive entries with the same speaker
    consolidated_transcriptions = []
    current_speaker = None
    current_entry = None
    # Iterate over all files in the folder
    print(len(file_names))
    for i, file_name in enumerate(file_names):
        print(i)
        if file_name.endswith(".wav"):
            # Construct the full path to the audio file
            audio_file_path = os.path.join(input_folder_path, file_name)

            # Extract speaker information from the file name
            speaker_info = file_name.split("-")[-1].replace(".wav", "")

            # Load the audio file
            waveform, original_sample_rate = torchaudio.load(audio_file_path)

            # Resample
            target_sample_rate = 16000  # Whisper model's sample rate
            resample_transform = Resample(original_sample_rate, target_sample_rate)
            waveform = resample_transform(waveform)

            # Split the audio into chunks with a certain duration
            chunk_size_seconds = 10
            chunk_size_samples = int(target_sample_rate * chunk_size_seconds)

            # Create a list to store information for each chunk
            chunk_info_list = []

            # Extract start time and duration from the file name
            start_time = float(file_name.split("start_")[1].split("-")[0])
            duration = float(file_name.split("duration_")[1].split("-")[0])

            # Calculate end time
            end_time = start_time + duration
            transcription_total = ""
            exec_time_file = 0

            for i in range(0, waveform.size(1), chunk_size_samples):
                chunk = waveform[:, i:i + chunk_size_samples]

                # Process the audio chunk
                start_exec_time = time.time()
                inputs = processor(chunk.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
                generated_ids = model.generate(**inputs)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                exec_time = time.time() - start_exec_time
                exec_times_list.append(exec_time)
                transcription_total += transcription + " "
                exec_time_file += exec_time

            # Append information to the list
            chunk_info_list.append({
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker_info,
                "transcription": transcription_total,
                "execution_time": exec_time_file
            })

            # Add the list of chunk information to the dictionary
            transcriptions_dict[file_name] = chunk_info_list

    # for file_name, chunk_info_list in transcriptions_dict.items():
    #     for chunk_info in chunk_info_list:
    #         if chunk_info["speaker"] == current_speaker:
    #             # Merge consecutive entries with the same speaker
    #             current_entry["end_time"] = chunk_info["end_time"]
    #             current_entry["transcription"] += " " + chunk_info["transcription"]
    #             current_entry["execution_time"] += chunk_info["execution_time"]
    #         else:
    #             # Start a new entry for a different speaker
    #             current_speaker = chunk_info["speaker"]
    #             current_entry = {
    #                 "start_time": chunk_info["start_time"],
    #                 "end_time": chunk_info["end_time"],
    #                 "speaker": current_speaker,
    #                 "transcription_model": model_name,
    #                 "transcription": chunk_info["transcription"],
    #                 "execution_time": chunk_info["execution_time"]
    #             }
    #             consolidated_transcriptions.append(current_entry)

    # Output JSON file
    output_json_path = f"{output_folder_path}/transcription_{audio_name}_results_whisper_BASE_new"
    # Save the transcriptions' dictionary to a JSON file
    with open(f"{output_json_path}.json", "w") as json_file:
        json.dump(list(transcriptions_dict.values()), json_file, indent=2)
    output_txt = f"{output_folder_path}/execution_time_transcription_{audio_name}_results_whisper_BASE_new.txt"
    with open(output_txt, "w") as f:
        f.write(f"Total execution time for transcription: {sum(exec_times_list)}")
    print(f"Transcriptions saved to {output_json_path}")
    json2excel(output_json_path)
