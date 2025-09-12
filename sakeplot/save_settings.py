# utils/save_settings.py
import yaml
import settings

def save_settings_to_yaml(path="settings.yaml"):
    """
    Save settings from settings.py into a YAML file.
    
    Parameters
    ----------
    path : str, optional
        Path to output YAML file. Defaults to 'settings.yaml'.
    """

    config = {
        "exclude_categories": settings.exclude_categories,
        "fft_freq_range": settings.fft_freq_range,
        "fft_overlap": settings.fft_overlap,
        "fft_win": settings.fft_win,
        "file_index": settings.file_index,
        "file_index_verified": settings.file_index_verified,
        "file_power_mat": settings.file_power_mat,
        "file_power_mat_verified": settings.file_power_mat_verified,
        "freq_ranges": settings.freq_ranges,
        "freq_ratios": settings.freq_ratios,
        "mains_noise": settings.mains_noise,
        "norm_groups": settings.norm_groups,
        "normalize": settings.normalize,
        "outlier_threshold": settings.outlier_threshold,
        "outlier_window": settings.outlier_window,
        "paired": settings.paired,
        "power_area_mat": settings.power_area_mat,
        "psd_mat": settings.psd_mat,
        "settigs_path": settings.settigs_path,
        "summary_plot_type": settings.summary_plot_type,
    }

    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    return path
