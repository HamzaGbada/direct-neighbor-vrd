# TODO: - Prepare the utils (helper) to create the graphs
#       - Prepare CNN (UNET and VGG) and Word Embedding model
#       - The graph building process is independent to training
#       - Use pretrained (I train by my self) for wordEmbedding (use BERT)
#       - Graph building process is as follow:
#           * Build Direct neighbor
#           * Assign Feature vector to each node (use Unet CNN model for embedding)
#           * Assign edge angle
#       - Create multiple model for GNN