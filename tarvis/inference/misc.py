from typing import Optional, Tuple
import os
import math


def is_oom_error(exc: RuntimeError):
    return repr(exc).startswith("RuntimeError('CUDA out of memory.")


def split_by_job_id(n_elements: int, job_id_str: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Splits the given number of elements based on SLURM variables and provided job specification
    :param n_elements: total number of elements to process
    :param job_id_str: string showing split e.g. "2/4" means that there is a 4-way split and the current chunk will be
    the second.
    Either this, or `SLURM_ARRAY_TASK_ID` has to be set. Note that `SLURM_ARRAY_TASK_ID` should be 1-based and not
    0-based, so e.g. if you want to run 10 jobs then set '--array' in the sbatch file to '1-10' and  NOT '0-9'.
    :return: start and end indices for the split
    """
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        assert not job_id_str
        chunk_num = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1  # 1-based index to 0-based
        total_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    elif job_id_str:
        tokens = job_id_str.split("/")
        assert len(tokens) == 2, tokens
        chunk_num, total_chunks = int(tokens[0]) - 1, int(tokens[1])
        assert 0 <= chunk_num < total_chunks, f"chunk number ({chunk_num}) is invalid for total chunks ({total_chunks})"

    else:
        return 0, n_elements, 1

    vids_per_chunk = int(math.ceil(n_elements / float(total_chunks)))
    start_idx = vids_per_chunk * chunk_num
    end_idx = min(vids_per_chunk * (chunk_num + 1), n_elements)

    return start_idx, end_idx, chunk_num