import json
import os
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def read_bbox_and_words(path: Path):
    bbox_and_words_list = []

    with open(path, 'r', errors='ignore') as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue

            split_lines = line.split(",")

            bbox = np.array(split_lines[0:8], dtype=np.int32)
            text = ",".join(split_lines[8:])

            # From the splited line we save (filename, [bounding box points], text line).
            # The filename will be useful in the future
            bbox_and_words_list.append([path.stem, *bbox, text])

    dataframe = pd.DataFrame(bbox_and_words_list,
                             columns=['filename', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'line'],
                             dtype=np.int16)
    dataframe = dataframe.drop(columns=['x1', 'y1', 'x3', 'y3'])

    return dataframe


def read_entities(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)

    dataframe = pd.DataFrame([data])
    return dataframe


# Assign a label to the line by checking the similarity of the line and all the entities
def assign_line_label(line: str, entities: pd.DataFrame):
    line_set = line.replace(",", "").strip().split()
    for i, column in enumerate(entities):
        entity_values = entities.iloc[0, i].replace(",", "").strip()
        entity_set = entity_values.split()

        matches_count = 0
        for l in line_set:
            if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                matches_count += 1

            if (column.upper() == 'ADDRESS' and (matches_count / len(line_set)) >= 0.5) or \
                    (column.upper() != 'ADDRESS' and (matches_count == len(line_set))) or \
                    matches_count == len(entity_set):
                return column.upper()

    return "O"


def assign_labels(words: pd.DataFrame, entities: pd.DataFrame):
    max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # Value, index
    already_labeled = {"TOTAL": False,
                       "DATE": False,
                       "ADDRESS": False,
                       "COMPANY": False,
                       "O": False
                       }

    # Go through every line in $words and assign it a label
    labels = []
    for i, line in enumerate(words['line']):
        label = assign_line_label(line, entities)

        already_labeled[label] = True
        if (label == "ADDRESS" and already_labeled["TOTAL"]) or \
                (label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])):
            label = "O"

        # Assign to the largest bounding box
        if label in ["TOTAL", "DATE"]:
            x0_loc = words.columns.get_loc("x0")
            bbox = words.iloc[i, x0_loc:x0_loc + 4].to_list()
            area = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])

            if max_area[label][0] < area:
                max_area[label] = (area, i)

            label = "O"

        labels.append(label)

    labels[max_area["DATE"][1]] = "DATE"
    labels[max_area["TOTAL"][1]] = "TOTAL"

    words["label"] = labels
    return words


if __name__ == "__main__":
    train_path = '../../data/SROIE2019/train/'
    test_path = '../../data/SROIE2019/test/'


    bbox_train_path = train_path+"/box/"
    entities_train_path = train_path+"/entities/"

    bbox_test_path = test_path+"/box/"
    entities_test_path = test_path+"/entities/"
    # Train Loop
    for filename in os.listdir(bbox_train_path):
        bbox_file_path = bbox_train_path+filename
        entities_file_path = entities_train_path+filename

        bbox = read_bbox_and_words(path=Path(bbox_file_path))
        entities = read_entities(path=Path(entities_file_path))

        bbox_labeled = assign_labels(bbox, entities)
        # indexAge = bbox_labeled[bbox_labeled['label'] == 'O'].index
        # bbox_labeled.drop(indexAge, inplace=True)
        if not os.path.isdir("../../data/SROIE_CSV/train/"):
            os.makedirs("../../data/SROIE_CSV/train/")
        bbox_labeled.to_csv("../../data/SROIE_CSV/train/"+filename[:-4]+".csv")

    # Test Loop
    for filename in os.listdir(bbox_test_path):
        bbox_file_path = bbox_test_path+filename
        entities_file_path = entities_test_path+filename

        bbox = read_bbox_and_words(path=Path(bbox_file_path))
        entities = read_entities(path=Path(entities_file_path))

        bbox_labeled = assign_labels(bbox, entities)
        # indexAge = bbox_labeled[bbox_labeled['label'] == 'O'].index
        # bbox_labeled.drop(indexAge, inplace=True)
        if not os.path.isdir("../../data/SROIE_CSV/test/"):
            os.makedirs("../../data/SROIE_CSV/test/")
        bbox_labeled.to_csv("../../data/SROIE_CSV/test/"+filename[:-4]+".csv")
