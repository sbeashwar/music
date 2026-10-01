# Curated audio corpus

This directory is the repository's allowlisted audio-data boundary. Audio files
committed here are intentionally curated inputs for raga detection training,
evaluation, or reproducible test cases.

Do not place recordings, generated scales, synthesized output, downloaded
reference tracks, or temporary audio here. Keep those files in ignored runtime
directories such as `recording/`, `output/`, or `eval/test_audio/`.

The `gen/` subdirectory is reserved for reproducible generated audio and remains
ignored. New curated files should use a raga-specific directory or `Songs/`,
include non-personal metadata, and be reviewed like source code before commit.
