# ---- feature_extractor.py ---- #

import opensmile
import librosa
import numpy as np

SR = 16000
smile = opensmile.Smile(
    feature_set = opensmile.FeatureSet.eGeMAPSv02,
    feature_level = opensmile.FeatureLevel.Functionals
)

def opensmile_features(file):
    
    try:
        y, _ = librosa.load(file, sr = SR)
        
        if len(y) < 2048:
            return None
        
        features = smile.process_signal(y,SR)
        features_vector = features.reset_index(drop=True).iloc[0].to_dict()
        
        return features_vector
    except Exception as e:
        print(f"{e}")
        return None
    
def extract_opensmile_features(row):
    
    file = row["file"]
    
    full_vector = {}
    
    full_vector["file"] = file
    full_vector["label"] = row["emotion"]
    
    try:
        full_vector.update(opensmile_features(file))
        
        return full_vector
    except Exception as e:
        print(f"Extraction failed for {file}: {e}")
        return None