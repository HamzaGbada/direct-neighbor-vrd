from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, cropped_bbox, labels):
        self.cropped_bbox = cropped_bbox
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cropped_bbox[idx], self.labels[idx]

