import numpy as np
import pandas as pd

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks:
        return [labeled]
    else:
        return masks

def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def generate_submission(predictions, meta):
    image_ids, encodings = [], []
    for image_id, prediction in zip(meta['ImageId'].values, predictions):
        for mask in decompose(prediction):
            image_ids.append(image_id)
            encodings.append(' '.join(str(rle) for rle in run_length_encoding(mask > 128.)))
    submission = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encodings})
    return submission