# General Information

This repository contains the source code for a system that identifies chords and aligns them with beat positions, leveraging pre-trained models from the [crema](https://github.com/bmcfee/crema) and [madmom](https://github.com/CPJKU/madmom) libraries. 

It's an integral part of my Master's thesis and was developed during my involvement with the [MusiCoLab project](https://musicolab.hmu.gr/musicolab/news_en.php).

## Repository Structure

- **data**: Ground truth, chords, beats, and audio files for analysis.
- **output**: Analysis results (graphs, boxplots, histograms, and CSV tables).
- **utilities**: Scripts supporting `batchAnalysisEvaluation.py`.
- **AudioFilePaths.txt**: Audio file paths, used in `batchAnalysisEaluation.py`.
- **batchAnalysisEvaluation.py**: Generates analysis results in `output`. This script evaluates the performance of the system across multiple files, ensuring comprehensive analysis and validation.
- **chordBeatAnalysis.py**: A command-line script that extracts and aligns chords with beats. It functions as the server-side content analysis component of the [Play-Along-Together (PAT) application](https://musicolab.hmu.gr/apprepository/playalong3/) . Access the PAT application's source code [here](https://github.com/NeoklisMimidis/playalong3).
- **requirements.txt**: Lists dependencies for running the scripts.

## Installation and Usage

- **Python 3.7 Required**
- **Dependencies**: Install from `requirements.txt`.
