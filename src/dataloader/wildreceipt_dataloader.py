from typing import Any

from doctr.datasets.datasets import VisionDataset
from doctr.datasets.utils import convert_target_to_relative


class WILDRECEIPT(VisionDataset):
    TRAIN = ('https://download.openmmlab.com/mmocr/data/wildreceipt.tar', 'wildreceipt.tar')

    def __init__(self, train: bool = True, **kwargs: Any) -> None:
        url, filename = self.TRAIN
        super().__init__(url, filename, None, True, pre_transforms=convert_target_to_relative, **kwargs)

    #     if train:
    #         dataset = load_dataset("Theivaprakasham/wildreceipt")['train']
    #     else:
    #         dataset = load_dataset("Theivaprakasham/wildreceipt")['test']
    #
    #     CONTEXT_SIZE = 2
    #     EMBEDDING_DIM = 50
    #     self.words = [dataset[doc_index]["words"][bbox_index] for doc_index in range(len(dataset)) for bbox_index in
    #                   range(len(dataset[doc_index]["words"]))]
    #
    #     vocab = set(self.words)
    #     word_to_ix = {word: i for i, word in enumerate(vocab)}
    #
    #     model = torch.load('wild_50_embedding.pt')
    #     global_vectors = GloVe(name='6B', dim=EMBEDDING_DIM)
    #     tokenizer = get_tokenizer("basic_english")
    #
    #     self.images = [np.array(
    #         Image.open(os.path.join(dataset[doc_index]["image_path"])).crop(dataset[doc_index]["bboxes"][bbox_index]))
    #         for doc_index in range(len(dataset)) for bbox_index in range(len(dataset[doc_index]["bboxes"]))]
    #     self.words = [dataset[doc_index]["words"][bbox_index] for doc_index in range(len(dataset)) for bbox_index in
    #                   range(len(dataset[doc_index]["words"]))]
    #     self.img_labels = [dataset[doc_index]["ner_tags"][bbox_index] for doc_index in range(len(dataset)) for
    #                        bbox_index in range(len(dataset[doc_index]["ner_tags"]))]
    #     self.word_tensors = []
    #     for doc_index in range(len(dataset)):
    #         for word in dataset[doc_index]["words"]:
    #             try:
    #                 embedding = model.weight[word_to_ix[word.lower()]]
    #             except KeyError:
    #                 embedding = global_vectors.get_vecs_by_tokens([word.lower()])
    #             self.word_tensors.append(embedding)
    #
    # def __len__(self):
    #     print(len(self.img_labels), len(self.images), len(self.word_tensors))
    #     return len(self.img_labels)
    #
    # def __getitem__(self, idx):
    #     image = self.images[idx]
    #     transform = transforms.Compose([
    #         ToTensor()
    #     ])
    #     image = transform(image)
    #
    #     word_embed = self.word_tensors[idx]
    #
    #     label = self.img_labels[idx]
    #
    #     return image, word_embed, label
