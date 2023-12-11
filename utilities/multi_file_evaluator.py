# -*- coding: utf-8 -*-
"""
@author: Neoklis
Python 3.7
"""

import os
import numpy as np
import madmom
from madmom.evaluation.chords import *
import pandas as pd


class ChordEvaluationExtended(madmom.evaluation.EvaluationMixin):
    """
    Provide various chord evaluation scores.(Added extra metric categories)

    Parameters
    ----------
    detections : str
        File containing chords detections.
    annotations : str
        File containing chord annotations.
    name : str, optional
        Name of the evaluation object (e.g., the name of the song).

    """

    METRIC_NAMES = [
        ("root", "Root"),
        ("majmin", "MajMin"),
        ("majminbass", "MajMinBass"),
        ("sevenths", "Sevenths"),
        ("seventhsbass", "SeventhsBass"),
        ("segmentation", "Segmentation"),
        ("oversegmentation", "OverSegmentation"),
        ("undersegmentation", "UnderSegmentation"),
        # ('thirds', 'Thirds'), #TODO
        # ('thirdsbass', 'ThirdsBass'), #TODO
        # ('mirex', 'MIREX'), #TODO
        # ('jazz5', 'Jazz5'), #TODO
        ("triads", "Triads"),
        ("triadsbass", "TriadsBass"),
        ("tetrads", "Tetrads"),
        ("tetradsbass", "TetradsBass"),
    ]

    def __init__(self, detections, annotations, name=None, **kwargs):
        self.name = name or ""
        self.ann_chords = merge_chords(encode(annotations))
        self.det_chords = merge_chords(adjust(encode(detections), self.ann_chords))
        self.annotations, self.detections, self.durations = evaluation_pairs(
            self.det_chords, self.ann_chords
        )
        self._underseg = None
        self._overseg = None

    @property
    def length(self):
        """Length of annotations."""
        return self.ann_chords["end"][-1] - self.ann_chords["start"][0]

    @property
    def root(self):
        """Fraction of correctly detected chord roots."""
        return np.average(score_root(self.detections, self.annotations), weights=self.durations)

    @property
    def majmin(self):
        """
        Fraction of correctly detected chords that can be reduced to major
        or minor triads (plus no-chord). Ignores the bass pitch class.
        """
        det_triads = reduce_to_triads(self.detections)
        ann_triads = reduce_to_triads(self.annotations)
        majmin_sel = select_majmin(ann_triads)
        return np.average(score_exact(det_triads, ann_triads), weights=self.durations * majmin_sel)

    @property
    def majminbass(self):
        """
        Fraction of correctly detected chords that can be reduced to major
        or minor triads (plus no-chord). Considers the bass pitch class.
        """
        det_triads = reduce_to_triads(self.detections, keep_bass=True)
        ann_triads = reduce_to_triads(self.annotations, keep_bass=True)
        majmin_sel = select_majmin(ann_triads)
        return np.average(score_exact(det_triads, ann_triads), weights=self.durations * majmin_sel)

    @property
    def sevenths(self):
        """
        Fraction of correctly detected chords that can be reduced to a seventh
        tetrad (plus no-chord). Ignores the bass pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections)
        ann_tetrads = reduce_to_tetrads(self.annotations)
        sevenths_sel = select_sevenths(ann_tetrads)
        return np.average(score_exact(det_tetrads, ann_tetrads), weights=self.durations * sevenths_sel)

    @property
    def seventhsbass(self):
        """
        Fraction of correctly detected chords that can be reduced to a seventh
        tetrad (plus no-chord). Considers the bass pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections, keep_bass=True)
        ann_tetrads = reduce_to_tetrads(self.annotations, keep_bass=True)
        sevenths_sel = select_sevenths(ann_tetrads)
        return np.average(score_exact(det_tetrads, ann_tetrads), weights=self.durations * sevenths_sel)

    @property
    def undersegmentation(self):
        """
        Normalized Hamming divergence (directional) between annotations and
        detections. Captures missed chord segments.
        """
        if self._underseg is None:
            self._underseg = 1 - segmentation(
                self.det_chords["start"],
                self.det_chords["end"],
                self.ann_chords["start"],
                self.ann_chords["end"],
            )
        return self._underseg

    @property
    def oversegmentation(self):
        """
        Normalized Hamming divergence (directional) between detections and
        annotations. Captures how fragmented the detected chord segments are.
        """
        if self._overseg is None:
            self._overseg = 1 - segmentation(
                self.ann_chords["start"],
                self.ann_chords["end"],
                self.det_chords["start"],
                self.det_chords["end"],
            )
        return self._overseg

    @property
    def segmentation(self):
        """Minimum of `oversegmentation` and `undersegmentation`."""
        return min(self.undersegmentation, self.oversegmentation)

    @property
    def triads(self):
        """
        Correctly detected triads chords (plus no-chord). Ignores the bass
        pitch class.
        """
        det_triads = reduce_to_triads(self.detections)
        ann_triads = reduce_to_triads(self.annotations)
        return np.average(score_exact(det_triads, ann_triads), weights=self.durations)

    @property
    def triadsbass(self):
        """
        Correctly detected triads chords (plus no-chord). Considers the bass
        pitch class.
        """
        det_triads = reduce_to_triads(self.detections, keep_bass=True)
        ann_triads = reduce_to_triads(self.annotations, keep_bass=True)
        return np.average(score_exact(det_triads, ann_triads), weights=self.durations)

    @property
    def tetrads(self):
        """
        Correctly detected triads chords (plus no-chord). Ignores the bass
        pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections)
        ann_tetrads = reduce_to_tetrads(self.annotations)
        return np.average(score_exact(det_tetrads, ann_tetrads), weights=self.durations)

    @property
    def tetradsbass(self):
        """
        Correctly detected tetrads chords (plus no-chord). Considers the bass
        pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections, keep_bass=True)
        ann_tetrads = reduce_to_tetrads(self.annotations, keep_bass=True)
        return np.average(score_exact(det_tetrads, ann_tetrads), weights=self.durations)

    # =============================================================================
    #     # TODO add one move vocabulary that matches the JAAH: https://github.com/MTG/JAAH
    #     @property
    #     def jazz5(self):
    #         """
    #         Correctly detected tetrads chords (plus no-chord). Considers the bass
    #         pitch class.
    #         """
    #         """
    #         Fraction of correctly detected chords that can be reduced to a maj, dom7,
    #         min, dim, hdim7 (plus no-chord). Ignores the bass pitch class.
    #
    #         CAREFUL: maj class includes maj7, maj6 etc,
    #             dim class includes dim7 etc
    #
    #         Check:
    #             AUDIO-ALIGNED JAZZ HARMONY DATASET FOR AUTOMATIC
    #             CHORD TRANSCRIPTION AND CORPUS-BASED RESEARCH
    #         """
    #         det_tetrads = reduce_to_tetrads(self.detections, keep_bass=True)
    #         ann_tetrads = reduce_to_tetrads(self.annotations, keep_bass=True)
    #         # jazz5_sel = select_jazz5(ann_tetrads) # TODO!
    #         return np.average(score_exact(det_tetrads, ann_tetrads),
    #                           weights=self.durations * jazz5_sel)
    # =============================================================================

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        eval_string : str
            Evaluation metrics formatted as a human readable string.

        """
        ret = (
            "{}\n"
            "  Root: {:5.2f} MajMin: {:5.2f} MajMinBass: {:5.2f} "
            "Sevenths: {:5.2f} SeventhsBass: {:5.2f}\n"
            "  Triads: {:5.2f} TriadsBass: {:5.2f}"
            " Tetrads: {:5.2f} TetradsBass: {:5.2f}\n"
            "  Seg: {:5.2f} UnderSeg: {:5.2f} OverSeg: {:5.2f}".format(
                self.name,
                self.root * 100,
                self.majmin * 100,
                self.majminbass * 100,
                self.sevenths * 100,
                self.seventhsbass * 100,
                self.triads * 100,
                self.triadsbass * 100,
                self.tetrads * 100,
                self.tetradsbass * 100,
                self.segmentation * 100,
                self.undersegmentation * 100,
                self.oversegmentation * 100,
            )
        )

        return ret


class ChordSumEvaluationExtended(ChordEvaluationExtended):
    """
    Class for averaging Chord evaluation scores, considering the lengths
    of the pieces. For a detailed description of the available metrics,
    refer to ChordEvaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str, optional
        Name to be displayed.

    """

    # pylint: disable=super-init-not-called

    def __init__(self, eval_objects, name=None):
        self.name = name or "weighted mean for %d files" % len(eval_objects)

        self.annotations = np.hstack([e.annotations for e in eval_objects])
        self.detections = np.hstack([e.detections for e in eval_objects])
        self.durations = np.hstack([e.durations for e in eval_objects])

        un_segs = [e.undersegmentation for e in eval_objects]
        over_segs = [e.oversegmentation for e in eval_objects]
        segs = [e.segmentation for e in eval_objects]
        lens = [e.length for e in eval_objects]

        self._underseg = np.average(un_segs, weights=lens)
        self._overseg = np.average(over_segs, weights=lens)
        self._seg = np.average(segs, weights=lens)
        self._length = sum(lens)

    def length(self):
        """Length of all evaluation objects."""
        return self._length

    @property
    def segmentation(self):
        return self._seg


# %%


def evaluate_chords_extended(ref_chords_filepath, est_chords_filepath):
    """
    Evaluates the accuracy of estimated chord annotations against reference annotations for a single file.
    Uses an extended chord evaluation method for detailed metrics.

    Parameters:
    - ref_chords_filepath (str): Path to the reference chord annotation file.
    - est_chords_filepath (str): Path to the estimated chord annotation file.

    Returns:
    - Evaluation metrics object for the chord annotations.
    """

    # MADMOM loading
    chords_est = madmom.io.load_chords(est_chords_filepath)
    chords_ref = madmom.io.load_chords(ref_chords_filepath)

    # eval_metrics = madmom.evaluation.ChordEvaluation(chords_est, chords_ref)
    eval_metrics = ChordEvaluationExtended(chords_est, chords_ref)
    return eval_metrics


def evaluate_beats(ref_beats_filepath, est_beats_filepath):
    """
    Evaluates the accuracy of estimated beat annotations against reference annotations for a single file.

    Parameters:
    - ref_beats_filepath (str): Path to the reference beat annotation file.
    - est_beats_filepath (str): Path to the estimated beat annotation file.

    Returns:
    - Evaluation metrics object for the beat annotations.
    """

    # MADMOM loading
    beats_est = madmom.io.load_beats(est_beats_filepath)
    beats_ref = madmom.io.load_beats(ref_beats_filepath)

    eval_metrics = madmom.evaluation.BeatEvaluation(beats_est, beats_ref)
    return eval_metrics


def evaluate_multiple_files(reference_filepaths, estimated_filepaths, evaluation_function):
    """
    Evaluates multiple files using a specified evaluation function, matching estimated files with reference files.

    Parameters:
    - reference_filepaths (list): List of paths to reference annotation files.
    - estimated_filepaths (list): List of paths to estimated annotation files.
    - evaluation_function (function): Function to use for evaluation.

    Returns:
    - Tuple containing a list of evaluation results and a list of unmatched files.
    """
    evals = []
    not_matched_files = []

    for est_filepath in estimated_filepaths:
        # Extract the last three parts of the estimated file path for precise matching
        # (to avoid bugs where the same audio file name exist in different artists)
        est_parts = est_filepath.split(os.sep)[-3:]

        # Find the corresponding reference file
        ref_filepath = None
        for ref in reference_filepaths:
            if ref.split(os.sep)[-3:] == est_parts:
                ref_filepath = ref
                break  # Quit inner 'for' loop on first match

        # Append to not_matched_files if no matching reference file is found
        if ref_filepath is None:
            not_matched_files.append(est_filepath)
        else:
            # Evaluate and append the results
            eval_metrics = evaluation_function(ref_filepath, est_filepath)
            evals.append(eval_metrics)

    return evals, not_matched_files


# %%


def search_and_evaluate(ref_paths, est_paths, task, extra_info=""):
    """
    Searches for files and performs evaluation for a specified task ('Chords', 'ChordsAtBeats', or 'Beats').
    It handles file searching, evaluation, and summarizing the results.

    Parameters:
    - ref_paths (str): Path to search for reference files.
    - est_paths (str): Path to search for estimated files.
    - task (str): The task for evaluation ('Chords', 'ChordsAtBeats', or 'Beats').
    - extra_info (str): Additional information to include in the evaluation summary (optional).

    Returns:
    - Tuple containing summary evaluation results and a DataFrame of all file evaluations.
    """

    if task == "Chords" or task == "ChordsAtBeats":
        suffix = "lab"
        evaluation_function = evaluate_chords_extended
        summary_evaluation_class = ChordSumEvaluationExtended
    elif task == "Beats":
        suffix = "txt"
        evaluation_function = evaluate_beats
        summary_evaluation_class = madmom.evaluation.BeatMeanEvaluation

    reference_filepaths = madmom.utils.search_files(ref_paths, suffix=suffix, recursion_depth=3)
    estimated_filepaths = madmom.utils.search_files(est_paths, suffix=suffix, recursion_depth=3)

    all_evals, _ = evaluate_multiple_files(reference_filepaths, estimated_filepaths, evaluation_function)

    # ChordSumEvaluationExtended || BeatMeanEvaluation
    result = summary_evaluation_class(all_evals)
    print(f"\n\t{task} {extra_info} Evaluation:\n", result.tostring())

    # Creating a list of lists from all_evals, and then a DataFrame for it
    data = [list(eval.metrics.values()) for eval in all_evals]
    df_all_files = pd.DataFrame(data, columns=list(all_evals[0].metrics.keys()))

    # Changing the order of the columns for the task of chords
    if task == "Chords" or task == "ChordsAtBeats":
        df_all_files = df_all_files[
            [
                "root",
                "majmin",
                "majminbass",
                "sevenths",
                "seventhsbass",
                "triads",
                "triadsbass",
                "tetrads",
                "tetradsbass",
                "segmentation",
                "undersegmentation",
                "oversegmentation",
            ]
        ]
    elif task == "Beats":
        """
        global_information_gain: if only 1 file is evaluated, it is the same as information gain
            So, use instead the implementaion of madmom that calculates it correctly
            (all files will share the same global_information_gain)
        """
        df_all_files["global_information_gain"] = result.metrics["global_information_gain"]

    return result, df_all_files
