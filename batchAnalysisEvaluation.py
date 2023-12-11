# %%
"""
@author: Neoklis
Python 3.7
"""

# Import necessary libraries
import madmom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from utilities.multi_file_evaluator import search_and_evaluate
from utilities.multi_file_analysis_and_prediction import (
    export_chords_beats,
    export_chords_beats_at_various_percents,
)

# Set the style for seaborn plots
sns.set_style("whitegrid")

# Directory paths for output data
detections_dir = "output/detections"  # Directory for detected chords and beats
alignment_dir = "output/aligned"  # Directory for aligned data

# Generate alignment percentages
# Use a loop to create a list of alignment percentages from 0.25 to 0.60
alignment_percentages = []
percent = 0.25
while percent <= 0.6:
    alignment_percentages.append(round(percent, 2))
    percent += 0.01

# Alternatively, use predefined custom alignment percentages
# alignment_percentages = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48, 0.5, 0.55, 0.6]

# Flag to control the saving of output files (e.g., text, CSV, plots)
SAVE = True

# IMPORTANT NOTES
# 1) Change the current working directory to the script folder path (batchAnalysisEvaluation.py)
#    to successfully import modules from the 'utilities' package.
# 2) Ensure all necessary audio files are downloaded before running the analysis,
#    or alternatively, load statistics from saved .csv files at the end of this script.

# %% --------------------------------------PREDICTIONS--------------------------------------

# Directory of audio files
audio_file_paths = madmom.utils.search_files(r"./data/audio", "mp3", 3)


if SAVE:
    file = open("AudioFilePaths.txt", "w")

# Iterate through all files and execute analysis & export relative files
for audio_file_path in audio_file_paths:
    export_chords_beats(audio_file_path, detections_dir, skipIfExists=True)

    # Save paths to a file if SAVE is True
    if SAVE:
        file.write(audio_file_path + "\n")

if SAVE:
    file.close()

# %% --------------------------------------ALIGNMENT--------------------------------------


chords_file_paths_crema = madmom.utils.search_files(f"./{detections_dir}/chords_crema", "lab", 3)
# chords_file_paths_madmom = madmom.utils.search_files(f"./{detections_dir}/chords_madmom", "lab", 3)
beats_file_paths = madmom.utils.search_files(f"./{detections_dir}/beats", "txt", 3)

# Create
for percent in alignment_percentages:
    print(f"\n Alignment Percent: {percent}\n")
    alignement_dir = f"{alignment_dir}/chordsAtBeats_crema" + str(percent)
    export_chords_beats_at_various_percents(
        chords_file_paths_crema, beats_file_paths, percent, alignement_dir, skipIfExists=True
    )

    # alignement_dir = f"{alignment_dir}/chordsAtBeats_madmom" + str(percent)
    # export_chords_beats_at_various_percents(chords_file_paths_madmom, beats_file_paths, percent, alignement_dir,      skipIfExists=True)


# %% --------------------------------------EVALUATION--------------------------------------
# (NOTE: Also, a dataframe is created with the 12 metrics across models and alignment combinations)

# --- Search files (Folder & file paths) ---

# - Chords (Crema) -
ref_path_chords = r"./data/gt/chords"
est_path_chords = f"./{detections_dir}/chords_crema"

chords_crema, chords_df_all_crema = search_and_evaluate(ref_path_chords, est_path_chords, "Chords")


# Create initial (empty) - only metrics names - dataframe, and then add results from CREMA
df = pd.DataFrame(index=list(chords_crema.metrics.keys()))
df.index.name = "Metrics"
df["Crema"] = list(chords_crema.metrics.values())


# - Chords (Madmom) -
ref_path_chords = r"./data/gt/chords"
est_path_chords2 = f"./{detections_dir}/chords_madmom"

chords_madmom, chords_df_all_madmom = search_and_evaluate(ref_path_chords, est_path_chords2, "Chords")
df["Madmom"] = list(chords_madmom.metrics.values())


# - Beats -
ref_path_beats = r"./data/gt/beats"  # Available ground truth only for the beatles dataset (179 songs)
est_path_beats = f"./{detections_dir}/beats"

beats_mean, beats_df_all = search_and_evaluate(ref_path_beats, est_path_beats, "Beats")

# %%

#  - Chords at beats evaluation -

# Create dictionalry to store the idependant evaluation of all tracks in various percentages
alignments_all = {}

for percent in alignment_percentages:
    artist = ""
    # artist = "/TheBeatles" # for Specific artist evaluation

    # Usage example:
    ref_path_chords_at_beats = r"./data/gt/chords" + artist

    est_path_chords_at_beats = f"./{alignment_dir}/chordsAtBeats_crema" + str(percent) + artist
    a, aligned_df = search_and_evaluate(
        ref_path_chords_at_beats, est_path_chords_at_beats, "ChordsAtBeats", percent
    )
    df[f"CREMA{percent}"] = list(a.metrics.values())

    # Insert into dictionary
    alignments_all[str(percent)] = aligned_df

    # .. you could also try and with madmom for the chord model. In this thesis CREMA was preferred
    # est_path_chords_at_beats = f"./{alignment_dir}/chordsAtBeats_madmom" + str(percent) + artist
    # b, _ = search_and_evaluate(ref_path_chords_at_beats, est_path_chords_at_beats, "ChordsAtBeats", percent)
    # df[f'Madmom{percent}'] = list(b.metrics.values())

# %% Reduce the dataframe to retain only the most accurate alignments
# (This is achieved by calculating their average across all 12 metrics and then sorting them)

# Step 1: Select columns from 3rd to 38th
selected_df = df.iloc[:, 2:38]  # Ignore first 2 columns bcs they are the default models (CREMA-MADMOM)

# Step 2: Calculate the average of each column
column_averages = selected_df.mean()

# Step 3: Sort the columns by their average and get the top 3 df from the selected_df
top_3_df = selected_df[column_averages.sort_values(ascending=False).head(3).index.tolist()]

# Step 4: Combine these top 3 columns with the first 2 columns of the original DataFrame
final_df = pd.concat([df.iloc[:, :2], top_3_df], axis=1)

# (final_df of size 12x5)

# %% --------------------------------------VISUALIZATION--------------------------------------

# - Chords -

# 1) Pointplots of the Median
dataframes = {
    "CREMA": chords_df_all_crema,
    "MADMOM": chords_df_all_madmom,
    "CREMA 0.41": alignments_all["0.41"],
    "CREMA 0.43": alignments_all["0.43"],
    "CREMA 0.42": alignments_all["0.42"],
}

# Concatenate the DataFrames along axis=1 (columns), creating a MultiIndex for columns
results = pd.concat(dataframes, axis=1)

# Melt the DataFrame to long format
df_long = results.reset_index().melt(id_vars="index")

# Get the unique 'variable' level-2 names (equivalent to minor_axis in Panel)
unique_tabs = df_long["variable_1"].unique()

plt.figure(figsize=(9, 8))
for i, tab in enumerate(unique_tabs, 1):
    ax = plt.subplot(4, 3, i)
    tab_data = df_long[df_long["variable_1"] == tab]
    sns.pointplot(
        data=tab_data,
        x="value",
        y="variable_0",
        orient="h",
        join=False,
        palette="Paired",
        capsize=0.5,
        estimator=np.median,  # Use median instead of mean
    )

    plt.axhline(0.5, alpha=0.1, zorder=-1, color="k")
    plt.axhline(1.5, alpha=0.1, zorder=-1, color="k")
    plt.axhline(2.5, alpha=0.1, zorder=-1, color="k")
    plt.axhline(3.5, alpha=0.1, zorder=-1, color="k")

    ax.set_ylabel("")  # Remove the Y-axis label 'variable_0'
    plt.title(tab)

    if i % 3 != 1:
        plt.yticks([])
    else:
        labs = ax.yaxis.get_ticklabels()
        ax.yaxis.set_ticklabels(labs, ha="left")
        ax.yaxis.set_tick_params(pad=50)
plt.tight_layout()
if SAVE:
    plt.savefig("output/PointplotMedian_for_415Files__2ChordModels_3AlignedChordsAtBeats.png")
plt.show()


# 2) Histrograms
alignments_all["0.41"].hist(bins=20, layout=(4, 3), figsize=(8, 8))
plt.tight_layout()
if SAVE:
    plt.savefig("output/Histogram_for_415Files__Best_alignment.png")
plt.show()

(chords_df_all_crema - alignments_all["0.41"]).hist(bins=20, layout=(4, 3), figsize=(8, 8))
plt.tight_layout()
if SAVE:
    plt.savefig("output/DifferenceHistogram_for_415Files__btw_Best_alignment_and_CREMA.png")
plt.show()


# 3) Boxplots
alignments_all["0.41"].boxplot(figsize=(8, 9.5), fontsize=13, rot=35)
plt.tight_layout()
if SAVE:
    plt.savefig("output/Boxplot_for_415Files__Best_alignment.png")
plt.show()

# TODO
# Find out and remove the audio files that the model is not good at || as a result reducing the variation

# %%
# - Beats -
def plot_beats_results(data, beat_annotation, beat_detection, sr=44100, name="", FIGSIZE=(10, 4)):
    """
    Plot the waveform of an audio signal with overlaid beat annotations.
    """

    beat_annotation = madmom.io.load_beats(beat_annotation)
    beat_detection = madmom.io.load_beats(beat_detection)

    plt.figure(figsize=FIGSIZE)
    librosa.display.waveshow(x, sr=sr, alpha=0.6)
    plt.vlines(
        beat_annotation,
        1.1 * x.min(),
        1.1 * x.max(),
        label="ground_truth",
        color="r",
        linestyle=":",
        linewidth=2,
    )
    plt.vlines(beat_detection.T, 1.1 * x.min(), 1.1 * x.max(), label="predicted", color="b", linewidth=1)

    plt.vlines(0, -1, 1, color="r", linewidth=5)

    plt.legend(fontsize=12)
    plt.title(f"{name} | Waveform with beats annotations", fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Time (sec)", fontsize=13)
    plt.xlim(0, len(x) / sr)


audio, sr = librosa.load("./data/audio/TheBeatles/01_-_Please_Please_Me/05_-_Boys.mp3", sr=44100)
x = audio[0 : sr * 20]  # first 20 seconds

gt = "./data/gt/beats/TheBeatles/01_-_Please_Please_Me/05_-_Boys.txt"
pred = "./output/detections/beats/TheBeatles/01_-_Please_Please_Me/05_-_Boys.txt"


plot_beats_results(x, gt, pred, name="The Beatles: 05_-_Boys")
if SAVE:
    plt.savefig("output/Waveform with beats annotations.png")
plt.show()


# %%
# %% Save and load results to .csv,  to be accessible at any time
# %%
# %% - Save results locally

if SAVE:
    chords_df_all_crema.to_csv(f"{est_path_chords}.csv")
    chords_df_all_madmom.to_csv(f"{est_path_chords2}.csv")

    beats_df_all.to_csv(f"{est_path_beats}.csv")

    alignments_all["0.41"].to_csv(f"./{alignment_dir}/chordsAtBeats_crema0.41.csv")
    alignments_all["0.43"].to_csv(f"./{alignment_dir}/chordsAtBeats_crema0.43.csv")
    alignments_all["0.42"].to_csv(f"./{alignment_dir}/chordsAtBeats_crema0.42.csv")

    # final_df: Weighted Chord Symbol Recall for all 415 audio files:
    # a) CREMA b) MADMOM c,d,e) chords_at_beats_(041 | 043 | 042) [-aligned at crema chord model]
    final_df.to_csv("output/WCSR_Metrics_for_415_Audio_Files__2ChordModels_3AlignedChordsAtBeats.csv")

    # Tables
    alignments_all["0.41"].describe().to_csv("output/DescriptiveStats_415Files__Best_alignment.csv")

    chords_df_all_crema.describe().to_csv("output/DescriptiveStats_415Files__ChordsCREMA.csv")
    chords_df_all_madmom.describe().to_csv("output/DescriptiveStats_415Files__ChordsMADMOM.csv")
    beats_df_all.describe().to_csv("output/DescriptiveStats_179Files__Beats.csv")
    # NOTE: beats == beats_df_all.describe().loc["mean"]


# %% - Load statistics from saved .csv

chords_df_all_crema = pd.read_csv(f"./{detections_dir}/chords_crema.csv", index_col=0)
chords_df_all_madmom = pd.read_csv(f"./{detections_dir}/chords_madmom.csv", index_col=0)

beats_df_all = pd.read_csv(f"./{detections_dir}/beats.csv", index_col=0)

alignments_all = {}
alignments_all["0.41"] = pd.read_csv(f"./{alignment_dir}/chordsAtBeats_crema0.41.csv", index_col=0)
alignments_all["0.43"] = pd.read_csv(f"./{alignment_dir}/chordsAtBeats_crema0.43.csv", index_col=0)
alignments_all["0.42"] = pd.read_csv(f"./{alignment_dir}/chordsAtBeats_crema0.42.csv", index_col=0)

final_df = pd.read_csv("output/WCSR_Metrics_for_415_Audio_Files__2ChordModels_3AlignedChordsAtBeats.csv")
