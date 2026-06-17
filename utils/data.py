import numpy as np
import pandas as pd


# ### feature generation

def generate_features(data, args): # [N, L] -> [N, L, F]
    features = []
    feature_cnt = 0
    if args.feature_separating_periods:
        feature_cnt += 1
        feature_fn = feature_separating_periods
        features.append(np.array([apply_feature_fn(series, feature_fn, args.trim_leading_zeros) for series in data]))
    
    if args.feature_successive_zero_periods:
        feature_cnt += 1
        feature_fn = feature_successive_zero_periods
        features.append(np.array([apply_feature_fn(series, feature_fn, args.trim_leading_zeros) for series in data]))
    
    features = stack_features(features, data)

    return features, feature_cnt

def apply_feature_fn(series, feat_fn, trim_leading_zeros=False):
    if trim_leading_zeros:
        return generate_feature_from_trimmed_leading_zeros(series, feat_fn)
    else:
        return feat_fn(series)
    
def generate_feature_from_trimmed_leading_zeros(series, feat_fn):
    first_non_zero = np.argmax(series != 0)
    if first_non_zero == 0:
        return feat_fn(series)
    else:
        trimmed_series = series[first_non_zero:]
        features = feat_fn(trimmed_series)

        padded_features = np.full(len(series), -1, dtype=float)
        padded_features[first_non_zero:] = features
        return padded_features
    
def stack_features(features_raw, data_raw):
    if len(features_raw) > 1:
        features = np.stack(features_raw, axis=2)
    elif len(features_raw) == 1:
        features = np.expand_dims(features_raw[0], axis=2)
    else:
        features = np.expand_dims(np.zeros_like(data_raw), -1)

    # features shaped [N, L, F]
    return features

# used in Gutierrez et al. 2008
def feature_separating_periods(series):
    features = np.zeros(len(series))

    last_nonzeros = []  # will store indices of last two nonzero elements

    for i, elem in enumerate(series):
        if elem != 0:
            last_nonzeros.append(i)
            # keep only the last two
            if len(last_nonzeros) > 2:
                last_nonzeros.pop(0)

        # compute feature
        if len(last_nonzeros) < 2:
            features[i] = 0
        else:
            i1, i2 = last_nonzeros
            features[i] = i2 - i1 - 1

    return features


# used in Mukhopadhyay et al. 2012
def feature_successive_zero_periods(series):
    features = np.zeros(len(series))

    accumulating_zeros = 0

    for i, elem in enumerate(series):
        if elem == 0:
            accumulating_zeros += 1
            features[i] = accumulating_zeros
        else:
            accumulating_zeros = 0

    return features


# ### series filters

def filter_series(args, data):
    df = describe_demand_characteristics(data, trim_leading_zeros=args.trim_leading_zeros)

    if args.filter_intermittent_lumpy:
        df = df[(df["SBC"] == "intermittent") | (df["SBC"] == "lumpy")]

    return df.index


def describe_demand_characteristics(data, trim_leading_zeros=False):
    N, L = data.shape

    results = []

    for series in data:
        if trim_leading_zeros:
            first_non_zero = np.argmax(series != 0)
            series = series[first_non_zero:]
        
        series_len = len(series)
        nonzero_idx = np.where(series > 0)[0]
        demand_values = series[nonzero_idx]
        num_nonzero = len(nonzero_idx)

        occurrence_rate = num_nonzero / series_len

        demand_mean = np.mean(demand_values) if num_nonzero > 0 else 0
        demand_std = np.std(demand_values) if num_nonzero > 0 else 0

        demand_per_period = np.sum(series) / series_len
        cv2 = (demand_std / demand_mean) ** 2 if demand_mean > 0 else 0
        adi = series_len / num_nonzero if num_nonzero > 0 else np.inf

        intervals = np.diff(np.concatenate(([-1], nonzero_idx))) if num_nonzero > 0 else np.array([])
        interval_mean = np.mean(intervals) if len(intervals) > 0 else 0
        interval_std = np.std(intervals) if len(intervals) > 0 else 0


        results.append([
            cv2, adi, get_SBC_class(cv2, adi), demand_mean, demand_std, interval_mean, interval_std, num_nonzero, occurrence_rate, demand_per_period, series_len
        ])

    df = pd.DataFrame(results, columns = [
        "CV2", "ADI", "SBC",
        "Demand Value Mean", "Demand Value Std",
        "Interval Len Mean", "Interval Len Std",
        "Demand Occurrences",
        "Demand Occurrence Rate", "Demand Per Period",
        "Series Length",
    ])

    return df

def get_SBC_class(cv2, adi):
    if cv2 < 0.49 and adi < 1.32:
        sbc_class = "smooth"
    elif cv2 < 0.49 and adi >= 1.32:
        sbc_class = "intermittent"
    elif cv2 >= 0.49 and adi >= 1.32:
        sbc_class = "lumpy"
    elif cv2 >= 0.49 and adi < 1.32:
        sbc_class = "erratic"

    return sbc_class