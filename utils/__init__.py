def select_evenly_spaced_checkpoints(filenames, num_select):
    """Return num_select evenly spaced checkpoint filenames sorted by iteration.

    Assumes: all filenames match '<ITER>backbone.pth' and 0 < num_select <= len(filenames).
    """
    import re, numpy as _np
    pattern = re.compile(r"^(\d+)backbone\.pth$")
    parsed = [(int(pattern.match(f).group(1)), f) for f in filenames]
    parsed.sort(key=lambda x: x[0])
    files_sorted = [p[1] for p in parsed]
    if num_select >= len(files_sorted):
        return files_sorted
    indices = _np.linspace(0, len(files_sorted) - 1, num=num_select, dtype=int)
    indices = sorted(set(indices))
    while len(indices) < num_select:
        for i in range(len(files_sorted)):
            if i not in indices:
                indices.append(i)
                if len(indices) == num_select:
                    break
    indices.sort()
    return [files_sorted[i] for i in indices]