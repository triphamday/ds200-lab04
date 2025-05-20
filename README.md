# Pokemon Classification with Streaming Data using Socket

## Project Overview

This project demonstrates a big data analysis pipeline by simulating a streaming data environment using socket communication. 

- **Producer Server** sends Pokemon images data over a socket connection.
- **Consumer Server** receives the streamed data and uses it to train a convolutional neural network (CNN) model for Pokemon image classification.
- The dataset used consists of 7,000 hand-cropped and labeled Pokemon images from [Kaggle Pokemon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data).
- The project supports incremental training by streaming data in batches, enabling continuous model updates.

## Project Structure
```
.
├── data/
│ ├── data_loader.py # Data loading utilities
│ └── parse_stream.py # Parsing streamed data
├── model/
│ ├── CNN.py # CNN model architecture
│ └── trainer.py # Model training script
├── producer/
│ ├── custom_dataset.py # Custom dataset loader for Pokemon images
│ └── producer.py # Producer server that streams data via socket
├── scripts/
│ ├── run_producer.sh # Script to start the producer server
│ ├── run_stream.sh # Script to start the streaming consumer
│ └── run_train.sh # Script to start the training process
├── output/
│ ├── checkpoint/ # Saved model checkpoints and metadata
│ └── _spark_metadata/ # Metadata for checkpointing
├── accuracy.py # Accuracy evaluation script
├── main.py # Main entry point to run the consumer & training
├── requirements.txt # Python dependencies
└── README.md # This file
```


## How to Run

### 1. Setup environment

Install required Python packages:

```
pip install -r requirements.txt
```

### 2. Run the Producer Server
This server streams batches of Pokemon images to the consumer.
```
bash scripts/run_producer.sh
```

### 3. Run the Consumer & Training
This will start the consumer server that receives data and trains the CNN model incrementally.
```
bash scripts/run_stream.sh
```
or directly run the main script:
```
python main.py
```
