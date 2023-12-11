# %%
"""
@author: Neoklis
Python 3.7
"""
import os
import time
import librosa
import numpy as np
import jams
import madmom
import argparse
from crema.analyze import analyze
import concurrent.futures


# %% Chords analysis (available methods: crema or madmom)


def extract_chords(data, sr, method="crema"):
    """
    Extracts chords from an audio signal using either 'crema' or 'madmom' method.
    'crema' uses "Structured training for large vocabulary chord recognition".
    'madmom' provides options for chord recognition using Deep Chroma Processor and
    Fully Convolutional Deep Auditory Model with Conditional Random Field.

    Parameters:
    - data: Audio signal for chord extraction.
    - sr: Sampling rate of the audio signal.
    - method: Chord recognition method ('crema' or 'madmom').

    Returns:
    - Chord intervals and labels.
    """

    chord_intervals = []
    chord_labels = []

    # CREMA : 'convolutional and recurrent estimators for music analysis'
    if method == "crema":
        # Chord recognition algorithm based on "Structured training for large vocabulary chord recognition"
        jams_crema = analyze(y=data, sr=sr)
        # Get the first chord annotation
        chord_annotation = jams_crema.search(namespace="chord")[0]

        chord_intervals, chord_labels = chord_annotation.to_interval_values()

    # MADMOM
    elif method == "madmom":
        # *CHORDS* (# fps=10 || meaning: in 44100 sample_rate -> 4410 frame_size || 100 ms btw frames)

        # =============================================================================
        #         #  Deep Chroma + Conditional Random Field (the fastest chord processor)
        #         dcp = madmom.audio.chroma.DeepChromaProcessor(fmin=65, fmax=2100, unique_filters=True)
        #         decode = madmom.features.DeepChromaChordRecognitionProcessor(fps=10)
        #         chords_processor_DCP = madmom.processors.SequentialProcessor([dcp, decode])
        #
        #         chords = chords_processor_DCP(data)
        # =============================================================================

        #  Fully Convolutional Deep Auditory Model + Conditional Random Field
        chord_features = madmom.features.CNNChordFeatureProcessor()
        chord_decoder = madmom.features.CRFChordRecognitionProcessor(fps=10)
        chords_processor_CNN = madmom.processors.SequentialProcessor([chord_features, chord_decoder])

        chords = chords_processor_CNN(data)

        for obs in chords:
            chord_intervals.append(np.array([obs["start"], obs["end"]]))
            chord_labels.append(obs["label"])
        chord_intervals = np.array(chord_intervals)

    else:
        print("Choose a valid method for chord recognition: crema or madmom")

    return chord_intervals, chord_labels


def extract_beats(data):
    """
    Extracts beats and downbeats from an audio signal using RNNs and HMM.

    Parameters:
    - data: Audio signal for beat extraction.

    Returns:
    - Extracted beats.
    """

    # joint beat and downbeat activation function from multiple RNNs + DBN approximated by a HMM
    downbeat_activation = madmom.features.RNNDownBeatProcessor()
    downbeat_tracker = madmom.features.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=100, transition_lambda=100
    )

    beats_processor_joint = madmom.processors.SequentialProcessor([downbeat_activation, downbeat_tracker])
    # the above model extracts beats & downbeats

    # Run
    beats = beats_processor_joint(data)

    return beats


def add_chords_to_beats(beats_times, chord_intervals, chord_labels, percent=0.5):
    """
    Matches chords to beat positions in an audio file. Creates a JAMS annotation object
    with chords aligned to beat positions based on a specified percentage of the beat duration.

    Parameters:
    - beats_times: Array of beat timing positions.
    - chord_intervals: Array of chord intervals (start and end times).
    - chord_labels: List of chord labels.
    - percent: Percentage of beat duration to match a chord (default 0.5).

    Returns:
    - JAMS annotation object with matched chords at beats.
    """

    # Instantiate a JAMS annotation object and populate it with the necessary contents
    annotation_object = jams.Annotation(namespace="chord")

    # Insert as last row a beat position matching the duration of the audio file
    beats_times = np.append(beats_times, chord_intervals[-1][-1])

    chord_idx = 0

    def limit_and_round(number):
        number = float(number)
        # Limit the number between 0 and 1
        if number < 0:
            number = 0
        elif number > 1:
            number = 1

        # Round the number to two decimal places
        return round(number, 2)

    # For every beat possition
    for beat_idx in range(len(beats_times) - 1):

        curr_beat_time = beats_times[beat_idx]
        next_beat_time = beats_times[beat_idx + 1]
        duration = next_beat_time - curr_beat_time

        # Decisive point for choosing appropriate chord
        next_chord_matching_percentage = duration * limit_and_round(percent)

        # Find the chord that matches the current beat position
        while chord_idx < len(chord_intervals):
            chord_start_time = chord_intervals[chord_idx][0]

            # Find the chord, that starts after the next_chord_matching_percentage of current beat
            if chord_start_time >= curr_beat_time + next_chord_matching_percentage:
                break  # break from the 'inner' "while loop" and continue with the 'outer' "for beat_idx loop"

            chord_idx += 1
        # end while chord_idx

        chord_prev_label = "N" if chord_idx == 0 else chord_labels[chord_idx - 1]
        matched_chord = [curr_beat_time, duration, str(chord_prev_label)]

        # Append in the JAMS object the aligned chords at beat positions
        annotation_object.append(time=matched_chord[0], duration=matched_chord[1], value=matched_chord[2])
    # end for beat_idx

    return annotation_object


# %%


def main(audio_file_path, output_path="", alignment_percent=0.41, title="", artist="", method="crema"):
    """
    Processes an audio file to extract and align chords to beats, and saves the result as a JAMS file.
    Uses either 'crema' or 'madmom' methods for chord extraction and madmom for beat tracking.
    Allows specifying metadata such as title and artist.

    Parameters:
    - audio_file_path: Path to the input audio file.
    - output_path: Directory to save the output JAMS file. Defaults to the same directory as the input file.
    - alignment_percent: Percentage of beat duration for aligning chords (default 0.41).
    - title: Title of the track (optional).
    - artist: Artist of the track (optional).
    - method: Method for chord extraction ('crema' or 'madmom', default 'crema').
    """

    start_time = time.time()

    # =============================================================================
    # # Output file name
    # =============================================================================

    # Extract the original file name
    file_name = os.path.basename(audio_file_path)

    # Remove the file extension
    file_name_without_ext = os.path.splitext(file_name)[0]

    # Print the output filename
    print(f"Analyzing {file_name_without_ext} ...")

    output_file_path = ""
    if output_path == "":
        # same dir as the given audio file
        output_file_path = f"{os.path.splitext(audio_file_path)[0]}.jams"
    else:
        # dir from the given output_path argument
        output_file_path = os.path.join(output_path, f"{file_name_without_ext}.jams")

    print(f"Output file path {output_file_path}")

    # =============================================================================
    # # Analysis
    # =============================================================================

    # Load an audio file with specific samplerate (& also librosa supports mutiple audio formats)
    samplerate = 44100
    data, sr = librosa.load(audio_file_path, sr=samplerate)

    # Run extract_chords and extract_beats concurrently
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit to the executor for concurrent execution
        future_chords = executor.submit(extract_chords, data, sr, method)
        future_beats = executor.submit(extract_beats, data)

        chord_intervals, chord_labels = future_chords.result()
        beats_times = future_beats.result().T[0]  # Take only beats information (no downbeats)

    # Create a JAMS annotation object and add a chord label for each beat in the audio file
    chords_at_beats = add_chords_to_beats(beats_times, chord_intervals, chord_labels, alignment_percent)

    # Include supplementary information regarding the annotation
    # this MUST always be 'program' because it will be refered as (automatic analysis) in the web application
    chords_at_beats.annotation_metadata.data_source = "program"

    if method == "crema":
        chords_at_beats.annotation_metadata.annotation_tools = (
            "MADMOM " + madmom.__version__ + " and CREMA 0.2.0"
        )

        chord_research_submission = "https://github.com/bmcfee/crema"

    elif method == "madmom":
        chords_at_beats.annotation_metadata.annotation_tools = "MADMOM " + madmom.__version__

        chord_research_submission = "https://madmom.readthedocs.io/en/v0.16.1/modules/features/chords.html"

    chords_at_beats.sandbox = {
        "description": f"Beat tracking uses the madmom implementation described in (1), while chord recognition relies on the algorithm outlined in (2).\n\n1.https://madmom.readthedocs.io/en/v0.16.1/modules/features/downbeats.html\n2.{chord_research_submission}"
    }

    # Create a Top-level JAMS object || this will be sent from the server to the client
    jams_musicolab = jams.JAMS()

    # Adding additional metadata about the file (one file can contain multiple annotations)
    jams_musicolab.file_metadata.duration = chord_intervals[-1][-1]
    jams_musicolab.file_metadata.title = title
    jams_musicolab.file_metadata.artist = artist

    # calculate process time and add information in jams file
    end_time = time.time()
    jams_musicolab.sandbox = {"analysis process time": f"{round(end_time - start_time, 2)} seconds"}

    # Append the JAMS annotation object
    jams_musicolab.annotations.append(chords_at_beats)

    # Save the filein the same directory as the file
    jams_musicolab.save(output_file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chord and beat analysis on audio file")

    parser.add_argument("inputfile", type=str, help="a full path to the input file")
    parser.add_argument("--output_path", type=str, default="", help="output path to save")
    parser.add_argument(
        "--alignment_percent",
        type=str,
        default="0.41",
        help="alignment percentage to be used. Values between 0.00 to 1.00",
    )
    parser.add_argument("--title", type=str, default="", help="title of the track")
    parser.add_argument("--artist", type=str, default="", help="artist of the track")
    parser.add_argument(
        "--method",
        type=str,
        default="crema",
        help='method to be used for the chord extraction Choose: "crema" or "madmom" (default: "crema")',
    )

    args = parser.parse_args()

    main(args.inputfile, args.output_path, args.alignment_percent, args.title, args.artist, args.method)
