import json
import matplotlib.pyplot as plt
import pandas as pd


def load_data(input_file):
    # Load the JSON data
    with open(input_file, "r") as json_file:
        data = json.load(json_file)
    return data


# computes how many seconds a speaker talks in the entire discussion
def compute_speaker_time(speaker, data):
    speaker_total_seconds = 0
    for entry in data:
        if entry["speaker"] == speaker:
            speech_time = entry["end_time"] - entry["start_time"]
            speaker_total_seconds += speech_time
    return speaker_total_seconds


# how much a speaker talks comparing to the entire discussion
def tra_p1_p2_len(speaker_total_seconds, data):
    total_discussion_time = data[-1]["end_time"]
    p1_p2_len_percent = speaker_total_seconds / total_discussion_time * 100
    return p1_p2_len_percent


def pol_poz(data, speaker1, speaker2):
    total_discussion_time_speaker1 = compute_speaker_time(speaker1, data)
    total_discussion_time_speaker2 = compute_speaker_time(speaker2, data)
    speaker1_pos = 0
    speaker2_pos = 0
    speaker1_neg = 0
    speaker2_neg = 0
    speaker1_neutral = 0
    speaker2_neutral = 0
    for entry in data:
        speech_time = entry["end_time"] - entry["start_time"]
        if entry["sentiment"] in ['no_impact', 'mixed']:  # neutral label
            if entry["speaker"] == speaker1:
                speaker1_neutral += speech_time
            elif entry["speaker"] == speaker2:
                speaker2_neutral += speech_time
        elif entry["sentiment"] == "positive":
            if entry["speaker"] == speaker1:
                speaker1_pos += speech_time
            elif entry["speaker"] == speaker2:
                speaker2_pos += speech_time
        elif entry["sentiment"] == "negative":
            if entry["speaker"] == speaker1:
                speaker1_neg += speech_time
            elif entry["speaker"] == speaker2:
                speaker2_neg += speech_time

    p1_poz = speaker1_pos / total_discussion_time_speaker1 * 100
    p1_neg = speaker1_neg / total_discussion_time_speaker1 * 100
    p1_neutral = speaker1_neutral / total_discussion_time_speaker1 * 100
    p2_poz = speaker2_pos / total_discussion_time_speaker2 * 100
    p2_neg = speaker2_neg / total_discussion_time_speaker2 * 100
    p2_neutral = speaker2_neutral / total_discussion_time_speaker2 * 100

    return {"P1_poz": p1_poz, "P1_neg": p1_neg, "P1_neutral": p1_neutral}, {"P2_poz": p2_poz, "P2_neg": p2_neg, "P2_neutral": p2_neutral}


def save_stats_to_xlsx(percentages_speech, percentages_sentiment_P1, percentages_sentiment_P2, output_filename):
    """
    :param percentages_speech:
    :param percentages_sentiment_P1:
    :param percentages_sentiment_P2:
    :param output_filename: filename.xlsx extension
    :return: None
    """
    # Define the column names
    column_names = ['Speech_time/discussion', 'Pol-Poz', 'Pol-Neg', 'Pol-Null']

    # Extract specific values from percentages_speech for each speaker
    speaker1_speech_value = percentages_speech["Speaker 1"]
    speaker2_speech_value = percentages_speech["Speaker 2"]

    # Create speaker data by concatenating speech values with sentiment values
    speaker1_data = [speaker1_speech_value] + list(percentages_sentiment_P1.values())
    speaker2_data = [speaker2_speech_value] + list(percentages_sentiment_P2.values())

    # Create a DataFrame with the data and column names
    df = pd.DataFrame([speaker1_data, speaker2_data], index=['Speaker 1', 'Speaker 2'], columns=column_names)

    # Write the DataFrame to an Excel file
    df.to_excel(output_filename)


def run_stats(input_filename, output_filename):
    data = load_data(input_filename)
    p1_time = compute_speaker_time("SPEAKER_00", data)
    p1_percent = tra_p1_p2_len(p1_time, data)
    percentages_speech = {"Speaker 1": p1_percent, "Speaker 2": 100 - p1_percent}
    percentages_sentiment_P1, percentages_sentiment_P2 = pol_poz(data, "SPEAKER_00", "SPEAKER_01")

    # Create a figure and three subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # Define colors for the speakers sentiments
    colors2 = ['green', 'red', 'grey']

    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''

    # Create first pie chart
    axs[0].pie(percentages_speech.values(), autopct=autopct_format, startangle=140)
    axs[0].set_title('Percentage of time each speaker talks Tra-P1/P2-Len')
    # Define colors for the speakers
    colors = ['blue', 'orange']
    # Create a color legend
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none') for color in colors]
    axs[0].legend(patches, percentages_speech.keys(), loc='upper left')

    # Creates second pie chart
    axs[1].pie(percentages_sentiment_P1.values(), autopct=autopct_format, startangle=140, colors=colors2)
    axs[1].set_title('Percentage of speaker 1 sentiments / discussion (Pol-P1)')

    # Create a color legend
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none') for color in colors2]
    axs[1].legend(patches, percentages_sentiment_P1.keys(), loc='upper left')

    # Creates second pie chart
    axs[2].pie(percentages_sentiment_P2.values(), autopct=autopct_format, startangle=140, colors=colors2)
    axs[2].set_title('Percentage of speaker 2 sentiments / discussion (Pol-P2)')

    # Create a color legend
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none') for color in colors2]
    axs[2].legend(patches, percentages_sentiment_P2.keys(), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    plt.savefig(output_filename.split(".")[0])
    # Show the plot
    # plt.show()

    save_stats_to_xlsx(percentages_speech, percentages_sentiment_P1, percentages_sentiment_P2, output_filename)

run_stats("emotion_results_p2_conv.json", "stats_p2_conv.xlsx")
