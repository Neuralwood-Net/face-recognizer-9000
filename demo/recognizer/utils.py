from operator import itemgetter
import numpy as np


def get_unique_predictions(probs, labels):
    pred_probs = [None for _ in range(len(probs))]
    pred_labels = [None for _ in range(len(probs))]

    available_labels = set(labels)
    available_labels.remove("Other")

    for img_idx, prob_dist in sorted(
        [(idx, prb) for idx, prb in enumerate(probs)],
        key=lambda pair: np.max(pair[1]),
        reverse=True,
    ):
        done = False
        for label_idx, prob in sorted(
            [(i, p) for i, p in enumerate(prob_dist)],
            key=itemgetter(1),
            reverse=True,
        ):
            current_label = labels[label_idx]

            if len(available_labels) == 0:
                pred_probs[img_idx] = prob_dist[3]
                pred_labels[img_idx] = "Other"
                done = True
                break

            elif current_label in available_labels:
                pred_probs[img_idx] = prob
                pred_labels[img_idx] = current_label
                available_labels.remove(current_label)
                break

            elif current_label == "Other":
                pred_probs[img_idx] = prob
                pred_labels[img_idx] = current_label
                break
        if done:
            break
    return pred_probs, pred_labels
