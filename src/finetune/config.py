config_dict={
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-07,
    "learning_rate": 5e-05,
    "early_stopping_steps": 5,
    "margin": 0.3,
    "loss_type": "triplet", # "cosine"
    "num_epochs": 1,

    "task_type": "train_eval",
    "train_batch_size": 1,
    "predict_batch_size": 2,

    "save_dir": "./test_finetune",
    "eval_model_dir": None,
    "model_name": "USE", # "USE", "albert", "bert" 
    "max_seq_length": 256,

    "kb_names": ["qa-pdpa"],
    "random_seed": 42,

    "MINIO_URL": "localhost:9001",
    "MINIO_ACCESS_KEY": "minio",
    "MINIO_SECRET_KEY": "minio123",
    "SAVEDMODELNAME": "finetuned.tar.gz"
    }

class config_obj:
    
    def __init__(self, config_dict=config_dict):
        self.__dict__.update(config_dict)
        self.config_dict = config_dict

    def __str__(self):
        return str(self.config_dict)
    
CONFIG=config_obj()
            