# -*- coding: utf-8 -*-
"""
@author: Neoklis
Python 3.7
"""

import os
import librosa
import numpy as np
import pandas as pd
from chordBeatAnalysis import extract_chords, extract_beats, add_chords_to_beats
import madmom
import concurrent.futures


# %%


def create_df_chords(chord_intervals, chord_labels):

    df = pd.DataFrame(chord_intervals, columns=["start_time", "end_time"])
    df["start_time"] = df["start_time"].round(3)
    df["end_time"] = df["end_time"].round(3)
    df["label"] = chord_labels
    return df


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def construct_file_path_OLD(base_output_path, sub_directory, file_name_without_ext, extension):
    path = os.path.join(base_output_path, sub_directory)
    ensure_directory_exists(path)
    return os.path.join(path, f"{file_name_without_ext}{extension}")


def construct_file_path(original_path, base_output_path, sub_directory, extension):
    parts = original_path.split(os.sep)
    artist, album, file_name = parts[-3], parts[-2], parts[-1]
    file_name_without_ext = os.path.splitext(file_name)[0]

    path = os.path.join(base_output_path, sub_directory, artist, album)
    ensure_directory_exists(path)
    return os.path.join(path, f"{file_name_without_ext}{extension}")


def file_exists(file_path):
    return os.path.isfile(file_path)


def load_chords_beats(chords_path, beats_path):

    # Chords loaded as arrays
    chords_est = madmom.io.load_chords(chords_path)
    chord_intervals = np.column_stack((chords_est["start"], chords_est["end"]))
    chord_labels = chords_est["label"]

    # Beats loaded as arrays
    beats_times = madmom.io.load_beats(beats_path)

    return beats_times, chord_intervals, chord_labels


# %%


def export_chords_beats(audio_file_path, base_output_path, skipIfExists=True):
    """
    Analyzes an audio file to extract chord and beat information and exports these to disk.

    Parameters:
    - audio_file_path (str): Path to the audio file for analysis.
    - base_output_path (str): Base directory for saving output files.
    - skipIfExists (bool): Skips processing if output files already exist (default: True).
    """

    # Construct file paths
    output_file_path_chords = construct_file_path(audio_file_path, base_output_path, "chords_crema", ".lab")
    output_file_path_chords2 = construct_file_path(
        audio_file_path, base_output_path, "chords_madmom", ".lab"
    )
    output_file_path_beats = construct_file_path(audio_file_path, base_output_path, "beats", ".txt")

    def execute_task(task, audio_file_path, skipIfExists):
        task_type, func, method, output_path = task

        # Load file only if analysis is required
        data, sr = librosa.load(audio_file_path, sr=44100)

        print(f"\n\tAnalyzing {task_type} for {audio_file_path} ...")
        if task_type in ["CREMA", "Madmom"]:
            chord_intervals, chord_labels = func(data, sr, method=method)
            df = create_df_chords(chord_intervals, chord_labels)
        else:  # Beats
            beats = func(data)
            df = pd.DataFrame(beats, columns=["beat_times", "beat_numbers"])

        df.to_csv(output_path, sep="\t", header=False, index=False)
        print(f"{task_type} analysis completed for {audio_file_path}. \nOutput saved to {output_path}")
        return f"{task_type} analysis completed for {audio_file_path}"

    # Define tasks for concurrent execution
    tasks = []
    if not (file_exists(output_file_path_chords) and skipIfExists):
        tasks.append(("CREMA", extract_chords, "crema", output_file_path_chords))

    if not (file_exists(output_file_path_chords2) and skipIfExists):
        tasks.append(("Madmom", extract_chords, "madmom", output_file_path_chords2))

    if not (file_exists(output_file_path_beats) and skipIfExists):
        tasks.append(("Beats", extract_beats, None, output_file_path_beats))

    # Using ProcessPoolExecutor to run the analysis tasks in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = [
            executor.submit(execute_task, task, audio_file_path, skipIfExists) for task in tasks
        ]
        for future in concurrent.futures.as_completed(future_results):
            results.append(future.result())
    if len(results) != 0:
        print(f"Completed analyses: {results}")


def export_chords_beats_at_various_percents(
    chords_file_paths, beats_file_paths, alignment_percent, alignment_dir, skipIfExists=True
):
    """
    Processes and exports aligned chord and beat data for multiple audio files across various alignment percentages.

    This function iterates over a list of chord file paths and beat file paths, aligning and exporting the chord data at beats  for each specified alignment percentage. It ensures that each chord file is matched with its corresponding beat file, aligns the chords at the specified beat percentages, and exports the aligned data to the specified output directory.

    Parameters:
    - chords_file_paths (list): List of paths to chord data files.
    - beats_file_paths (list): List of paths to beat data files.
    - alignment_percent (float): Percentage for aligning chords with beats.
    - alignment_dir (str): Directory for saving aligned data.
    - skipIfExists (bool): Skips processing if output already exists (default: True).
    """
    for chords_path in chords_file_paths:
        chords_file_name_without_ext = os.path.splitext(os.path.basename(chords_path))[0]

        for beats_path in beats_file_paths:
            beats_file_name_without_ext = os.path.splitext(os.path.basename(beats_path))[0]

            if chords_file_name_without_ext == beats_file_name_without_ext:
                # Construct file path for aligned chords
                output_file_path_chords_at_beats = construct_file_path(
                    chords_path, "", alignment_dir, ".lab"
                )

                # Check if file already exists and if skipping is enabled
                if skipIfExists and file_exists(output_file_path_chords_at_beats):
                    print("Skipped! Aligned detection exists for: ", chords_file_name_without_ext)
                else:
                    # Perform alignment and export
                    beats_times, chord_intervals, chord_labels = load_chords_beats(chords_path, beats_path)
                    chords_at_beatsJAMS = add_chords_to_beats(
                        beats_times, chord_intervals, chord_labels, alignment_percent
                    )

                    # Create dataframe and export
                    chord_intervals, chord_labels = chords_at_beatsJAMS.to_interval_values()
                    chords_at_beats_df = create_df_chords(chord_intervals, chord_labels)
                    chords_at_beats_df.to_csv(
                        output_file_path_chords_at_beats, sep="\t", header=False, index=False
                    )
                    print(f"Output file path:\n{output_file_path_chords_at_beats}")

                break  # Match found, no need to continue inner loop
        else:
            # This else corresponds to the for loop; executed when no break occurs (i.e., no match found)
            print(f"No exact match found for {chords_file_name_without_ext}")
