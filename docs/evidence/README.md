# CUDA benchmark evidence

`cuda-colab-t4-report-n15.json` is the current evidence. It records one discarded
warm-up fit followed by 15 timed CPU fits and 15 timed CUDA fits on a Colab Tesla
T4. Its `source_file_sha256` identifies the unedited report downloaded from
Colab. The verifier recomputes its timing statistics and permutation p-value
from the raw arrays.

`cuda-colab-t4-report.json` is retained as a historical, underpowered run. Its
five timed fits included startup effects and produced the superseded CUDA-faster
headline. The file's measured numbers are unchanged; `superseded_by` points
machine-readably to the n=15 replacement. The verifier returns `superseded` for
this specific retired artifact, while any underpowered report without a valid
replacement remains a hard error.

Neither benchmark evaluated the official test split.
