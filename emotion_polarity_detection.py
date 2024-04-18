import json


def run_emotion_model(pipe, input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in data:
        transcription = entry['transcription']
        sentiment = pipe(transcription)
        entry['sentiment'] = sentiment[0]["label"].lower()

    # Write the updated transcriptions with sentiment to a new JSON file
    output_path = output_file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print("one model done")