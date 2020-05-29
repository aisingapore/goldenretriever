"""
Finetune goldenretriever on knwoledge bases in Elasticsearch

Sample usage:
------------
python -m src.finetune.main
"""
import os
import pickle
import datetime
import pandas as pd
import numpy as np
import logging
import random
import sys
import tarfile
import shutil

import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import rankdata

from src.models import GoldenRetriever
from src.encoders import USEEncoder, ALBERTEncoder, BERTEncoder
from src.data_handler.kb_handler import kb, kb_handler
from src.minio_handler import MinioClient
from src.finetune.eval import eval_model
from src.finetune.generators import random_triplet_generator, hard_triplet_generator
from src.finetune.config import CONFIG

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # To prevent duplicate logs
    logger.propagate = False

    # Define file/directory paths
    concat_kb_names = "_".join(CONFIG.kb_names)

    model_folder_name = "model_" + concat_kb_names
    results_folder_name = "results_" + concat_kb_names
    MAIN_DIR = CONFIG.save_dir

    MODEL_DIR = os.path.join(MAIN_DIR, model_folder_name, CONFIG.model_name)
    MODEL_BEST_DIR = os.path.join(MAIN_DIR, model_folder_name, CONFIG.model_name, 'best')
    MODEL_LAST_DIR = os.path.join(MAIN_DIR, model_folder_name, CONFIG.model_name, 'last')
    EVAL_DIR = os.path.join(MAIN_DIR, results_folder_name, CONFIG.model_name)

    if not os.path.isdir(MODEL_LAST_DIR): os.makedirs(MODEL_LAST_DIR)
    if not os.path.isdir(EVAL_DIR):os.makedirs(EVAL_DIR)

    EVAL_SCORE_PATH = os.path.join(EVAL_DIR, '_eval_scores.xlsx')
    EVAL_DICT_PATH = os.path.join(EVAL_DIR, '_eval_details.pickle')

    logger.info(f'\nModels will be saved at: {MODEL_DIR}')
    logger.info(f'Best model will be saved at: {MODEL_BEST_DIR}')
    logger.info(f'Last trained model will be saved at {MODEL_LAST_DIR}')
    logger.info(f'Saving Eval_Score at: {EVAL_SCORE_PATH}')
    logger.info(f'Saving Eval_Dict at: {EVAL_DICT_PATH}\n')

    # Create training set based on chosen random seed
    logger.info("Generating training/ evaluation set")

    
    """
    LOAD MODEL
    """
    # Instantiate chosen model
    logger.info(f"Instantiating model: {CONFIG.model_name}")
    encoders = {
        "albert": ALBERTEncoder,
        "bert": BERTEncoder,
        "USE": USEEncoder
    }

    if CONFIG.model_name not in encoders:
        raise ValueError("Model not found: %s" % (CONFIG.model_name))

    # init the model and encoder
    enc = encoders[CONFIG.model_name](max_seq_length=CONFIG.max_seq_length)
    model = GoldenRetriever(enc)

    # Set optimizer parameters
    model.opt_params = {'learning_rate': CONFIG.learning_rate,'beta_1': CONFIG.beta_1,'beta_2': CONFIG.beta_2,'epsilon': CONFIG.epsilon}



    """
    PULL AND PARSE KB FROM SQL
    """
    train_dict = dict()
    test_dict = dict()
    df_list = []

    # Get df using kb_handler
    kbh = kb_handler()
    kbs = kbh.load_es_kb(CONFIG.kb_names)

    df = pd.concat([single_kb.create_df() for single_kb in kbs]).reset_index(drop='True')
    kb_names = df['kb_name'].unique()

    for kb_name in kb_names:
        kb_id = df[df['kb_name'] == kb_name].index.values
        train_idx, test_idx = train_test_split(kb_id, test_size=0.4,
                                            random_state=100)

        train_dict[kb_name] = train_idx
        test_dict[kb_name] = test_idx



    """
    FINETUNE
    """
    if CONFIG.task_type == 'train_eval':
        logger.info("Fine-tuning model")

        # see the performance of out of box model
        OOB_overall_eval, eval_dict = eval_model(model, df, test_dict)
        epoch_eval_score = OOB_overall_eval.loc['Across_all_kb','mrr_score']
        logger.info(f'Eval Score for OOB: {epoch_eval_score}')

        earlystopping_counter = 0
        for i in range(CONFIG.num_epochs):
            epoch_start_time = datetime.datetime.now()
            logger.info(f'Running Epoch #: {i}')

            cost_mean_total = 0
            batch_counter = 0
            epoch_start_time = datetime.datetime.now()

            # train_dataset_loader = random_triplet_generator(df, train_dict, CONFIG)
            train_dataset_loader = hard_triplet_generator(df, train_dict, model, CONFIG)

            for q, r, neg_r in train_dataset_loader:
                
                if random.randrange(100) <= 10:
                    logger.info(f'\nTRIPLET SPOT CHECK')
                    logger.info(f'{q[0]}')
                    logger.info(f'{r[0]}')
                    logger.info(f'{neg_r[0]}\n')
                
                batch_start_time = datetime.datetime.now()

                if batch_counter % 100 == 0:
                    logger.info(f'Running batch #{batch_counter}')
                cost_mean_batch = model.finetune(question=q, answer=r, context=r, \
                                                 neg_answer=neg_r, neg_answer_context=neg_r, \
                                                 margin=CONFIG.margin, loss=CONFIG.loss_type)

                cost_mean_total += cost_mean_batch

                batch_end_time = datetime.datetime.now()

                if batch_counter == 0 and i == 0:
                    len_training_triplets = sum([len(train_idxes) for kb, train_idxes in train_dict.items()])
                    num_batches = len_training_triplets // len(q)
                    logger.info(f'Training batches of size: {len(q)}')
                    logger.info(f'Number of batches per epoch: {num_batches}')
                    logger.info(f'Time taken for first batch: {batch_end_time - batch_start_time}')

                if batch_counter == num_batches:
                    break
                
                batch_counter += 1


            epoch_overall_eval, eval_dict = eval_model(model, df, test_dict)
            epoch_eval_score = epoch_overall_eval.loc['Across_all_kb','mrr_score']
            print(epoch_eval_score)

            logger.info(f'Number of batches trained: {batch_counter}')
            logger.info(f'Loss for Epoch #{i}: {cost_mean_total}')
            logger.info(f'Eval Score for Epoch #{i}: {epoch_eval_score}')

            epoch_end_time = datetime.datetime.now()
            logger.info(f'Time taken for Epoch #{i}: {epoch_end_time - epoch_start_time}')

            # Save model for first epoch
            if i == 0:
                lowest_cost = cost_mean_total
                highest_epoch_eval_score = epoch_eval_score
                best_epoch = i
                earlystopping_counter = 0
                best_model_path = os.path.join(MODEL_BEST_DIR, str(i))
                if os.path.exists(best_model_path) : shutil.rmtree(best_model_path)
                os.makedirs(best_model_path)
                model.export_encoder(best_model_path)

            # Model checkpoint
            if epoch_eval_score > highest_epoch_eval_score:
                best_epoch = i
                lowest_cost = cost_mean_total
                highest_epoch_eval_score = epoch_eval_score
                best_model_path = os.path.join(MODEL_BEST_DIR, str(i))
                if os.path.exists(best_model_path) : shutil.rmtree(best_model_path)
                os.makedirs(best_model_path)
                model.export_encoder(best_model_path)
                logger.info(f'Saved best model with cost of {lowest_cost} for Epoch #{i}')
                logger.info(f'Saved best model with cost of {highest_epoch_eval_score} for Epoch #{i}')
                earlystopping_counter = 0
            else:
                # Activate early stopping counter
                earlystopping_counter += 1

            # Early stopping
            if earlystopping_counter == CONFIG.early_stopping_steps:
                logger.info("Early stop executed")
                model.export_encoder(MODEL_LAST_DIR)
                break
            
            epoch_end_time = datetime.datetime.now()
            logger.info(f'Time Taken for Epoch #{i}: {epoch_end_time - epoch_start_time}')
            logger.info(f'Average time Taken for each batch: {(epoch_end_time - epoch_start_time)/batch_counter}')

    # Restore best model. User will have to define path to model if only eval is done.
    logger.info("Restoring model")
    if CONFIG.task_type == 'train_eval':
        model.restore_encoder(os.path.join(MODEL_BEST_DIR, str(best_epoch)))
    else:
        if CONFIG.eval_model_dir:
            model.restore_encoder(CONFIG.eval_model_dir)
        else:
            logger.info("Using out-of-box model")
            pass


    """
    EVAL MODEL
    """
    logger.info("Evaluating model")
       
    overall_eval, eval_dict = eval_model(model, df, test_dict)
    print("="*10 + ' OOB ' + "="*10)
    print(OOB_overall_eval)
    print("="*10 + ' FINETUNED ' + "="*10)
    print(overall_eval)

    # save the scores and details for later evaluation. WARNING: User will need to create the necessary directories to save df
    overall_eval.to_excel(EVAL_SCORE_PATH)
    with open(EVAL_DICT_PATH, 'wb') as handle:
        pickle.dump(eval_dict, handle)

    """
    SAVE MODEL IN MINIO
    """
    minio = MinioClient(CONFIG.MINIO_URL, CONFIG.MINIO_ACCESS_KEY, CONFIG.MINIO_SECRET_KEY)

    tar = tarfile.open("weights.tar.gz", mode="w:gz")
    tar.add( os.path.join(MODEL_BEST_DIR, str(best_epoch)) )
    tar.close()

    minio.make_bucket("finetunedweights")
    minio.upload_model_weights("finetunedweights",
                                CONFIG.SAVEDMODELNAME, 
                                "weights.tar.gz")
   
    os.remove("weights.tar.gz")
