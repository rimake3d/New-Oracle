import pandas as pd
from ragatouille import RAGTrainer

def load_data(pairs_filepath, documents_filepath):
    print("Loading data...")
    pairs_list = pd.read_csv(pairs_filepath)
    pairs = pairs_list.values.tolist()

    with open(documents_filepath, 'r') as file:
        raw_text = file.read()
    
    return pairs, raw_text

def prepare_and_train_model(pairs, raw_text):
    print("Loading model: colbert-ir/colbertv2.0")
    trainer = RAGTrainer(model_name="tax", pretrained_model_name="colbert-ir/colbertv2.0")

    print("Preparing training data...")
    trainer.prepare_training_data(raw_data=pairs, all_documents=raw_text, data_out_path='./data/', 
                                  num_new_negatives=10, hard_negative_minimum_rank=10, 
                                  mine_hard_negatives=True, hard_negative_model_size='small', 
                                  pairs_with_labels=False, positive_label=1, negative_label=0)
    print("Training data saved to ./data/")

    print("Starting training...")
    trainer.train(batch_size=32,
                  nbits=4,
                  maxsteps=500000,
                  use_ib_negatives=True,
                  dim=128,
                  learning_rate=5e-6,
                  doc_maxlen=256,
                  use_relu=False,
                  warmup_steps="auto")

def main():
    pairs_filepath = 'pairs.csv'
    documents_filepath = 'long_string.csv'

    pairs, raw_text = load_data(pairs_filepath, documents_filepath)
    prepare_and_train_model(pairs, raw_text)

if __name__ == "__main__":
    main()
