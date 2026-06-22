"""Crowd-sourced pairwise-comparison annotation collection.

Users rank the variants of a reference image (algorithm reconstructions + the
reference itself) by repeated two-alternative forced choice. A full ranking is
obtained with binary insertion sort (~n·log n comparisons), and the individual
pairwise outcomes are stored for later Bradley-Terry scoring.
"""
