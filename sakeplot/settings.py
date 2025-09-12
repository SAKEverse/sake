# settings.py

# Categories to exclude when plotting/analysis
exclude_categories = [
    "cat1",
    "cat2",
]

# Frequency range for FFT (Hz)
fft_freq_range = [1, 150]

# FFT parameters
fft_overlap = 0.5  # proportion of overlap between FFT windows
fft_win = 5        # FFT window size in seconds

# File names for different stages of processing
file_index = "index.csv"                    # raw index file
file_index_verified = "index_verified.csv"  # index file after verification
file_power_mat = "power_mat.pickle"         # power matrix (raw)
file_power_mat_verified = "power_mat_verified.pickle"  # power matrix after verification

# Frequency ranges of interest (Hz)
freq_ranges = [
    [2, 5],    # delta / low-theta
    [6, 12],   # theta / alpha
    [15, 30],  # beta
    [40, 70],  # gamma low
    [80, 120], # gamma high
]

# Frequency band ratios to compute (Hz pairs)
freq_ratios = [
    [[2, 5], [6, 12]],     # ratio of delta/theta to alpha
    [[15, 30], [40, 70]],  # ratio of beta to gamma
]

# Mains (powerline) noise frequencies to exclude (Hz)
mains_noise = [59, 61]

# Normalization options
norm_groups = ["", ""]  # [column, group] for normalization
normalize = 0           # 0 = disabled, 1 = enabled

# Outlier detection settings
outlier_threshold = 7   # threshold for identifying PSD outliers
outlier_window = 60     # time window for evaluating outliers (s)

# Whether analysis is paired (1 = paired, 0 = unpaired)
paired = 1

# Output files for different results
power_area_mat = "power_area.csv"  # power area results
psd_mat = "psd.csv"                # PSD results

# Path to the YAML settings file (used in cli.py to persist changes)
settigs_path = "settings.yaml"

# Default summary plot type (used in plot command)
summary_plot_type = "violin"
