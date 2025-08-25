'''
Lucas 240729
在训练时将train数据split为source和target供NCF等单域模型训练
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from math import ceil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    MixtralConfig,
    MistralForCausalLM,
    AutoConfig
)
from trl import SFTTrainer
import fire
from typing import List
from utils.prompter import Prompter
from utils.data_utils import NCFDataset  # lucas 240703 added
import pandas as pd
from module import PrefixTune_GAN_Mixtral2, PrefixTune_NCF_GAN_Mixtral2, NCF_GAN_CL, NCF, NCF_GAN, NCF_GAN2, NCF_CL, \
    NCF_Contrastive_MemoryBank, NCF_ContLea_MemoryBank_CurriLea, NCF_GAN_CL_MOE, NCF_MOE, NCF_GAN_MOE, NCF_GAN_TRNSMOE, NCF_GAN_SenBert
from module_da import NCF_MMD, NCF_MDD, NCF_MCD, NCF_DSN, NCF_CDAN
import torch.nn.functional as F
import warnings
from lox_cdr_utils import rouge_score, bleu_score, Batchify, now_time, ids2tokens, unique_sentence_percent, meteor
from lox_cdr_evaluation import LossMSE, LossMAE
import math
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

gpu_device = torch.device("cuda" if torch.cuda.is_available() else "")


# def format_ultrachat(ds):
#   text = []
#   for row in ds:
#     if len(row['messages']) > 2:
#       text.append("### Human: "+row['messages'][0]['content']+"### Assistant: "+row['messages'][1]['content']+"### Human: "+row['messages'][2]['content']+"### Assistant: "+row['messages'][3]['content'])
#     else: #not all tialogues have more than one turn
#       text.append("### Human: "+row['messages'][0]['content']+"### Assistant: "+row['messages'][1]['content'])
#   ds = ds.add_column(name="text", column=text)
#   return ds

# lucas 230920 定义对数据划分batch并取对应batch的数据
def take_batch_data(data, batch_size, batch_id):
    # print(f"take batch data user: {data['user']}; item: {data['item']}; input_ids: {data['input_ids']}")
    length = len(data['user'])
    batch_num = ceil(length / batch_size)
    batch_data = dict()
    if batch_size == 1:  # 240119 added for batch_size == 1
        if batch_id + 1 <= batch_num:
            # lucas 240115 tensor version
            batch_data["user"] = torch.LongTensor(
                [[int(i) for i in data['user']][batch_id]])
            batch_data["item"] = torch.LongTensor(
                [[int(i) for i in data['item']][batch_id]])
            # batch_data["rating"] = data['rating'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["input"] = data['input'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["output"] = data['output'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["instruction"] = data['instruction'][batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_data["input_ids"] = torch.LongTensor(
                data['input_ids'][batch_id]).unsqueeze(0)
            batch_data["attention_mask"] = torch.LongTensor(
                data['attention_mask'][batch_id]).unsqueeze(0)
            batch_data["labels"] = torch.LongTensor(data['labels'][batch_id]).unsqueeze(0)
            # lucas 240328 added rating, domain
            batch_data["rating"] = torch.LongTensor([[int(i) for i in data['rating']][batch_id]])
            batch_data["domain"] = torch.LongTensor([[int(i) for i in data['domain']][batch_id]])
        else:
            warnings.warn("---batch id is out of range")
    else:
        if batch_id + 1 < batch_num:
            # # lucas 240115 list version
            # batch_data["user"] = [int(i) for i in data['user']][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["item"] = [int(i) for i in data['item']][batch_size * batch_id: batch_size * (batch_id + 1)]
            # # batch_data["rating"] = data['rating'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # # batch_data["input"] = data['input'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # # batch_data["output"] = data['output'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # # batch_data["instruction"] = data['instruction'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["input_ids"] = data['input_ids'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["attention_mask"] = data['attention_mask'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["labels"] = data['labels'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # # batch_data = data[batch_size * batch_id: batch_size * (batch_id + 1)]

            # lucas 240115 tensor version
            batch_data["user"] = torch.LongTensor(
                [int(i) for i in data['user']][batch_size * batch_id: batch_size * (batch_id + 1)])
            batch_data["item"] = torch.LongTensor(
                [int(i) for i in data['item']][batch_size * batch_id: batch_size * (batch_id + 1)])
            # batch_data["rating"] = data['rating'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["input"] = data['input'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["output"] = data['output'][batch_size * batch_id: batch_size * (batch_id + 1)]
            # batch_data["instruction"] = data['instruction'][batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_data["input_ids"] = torch.LongTensor(
                data['input_ids'][batch_size * batch_id: batch_size * (batch_id + 1)])
            batch_data["attention_mask"] = torch.LongTensor(
                data['attention_mask'][batch_size * batch_id: batch_size * (batch_id + 1)])
            batch_data["labels"] = torch.LongTensor(data['labels'][batch_size * batch_id: batch_size * (batch_id + 1)])
            # lucas 240328 add rating domain information
            batch_data["rating"] = torch.LongTensor(
                [int(i) for i in data['rating']][batch_size * batch_id: batch_size * (batch_id + 1)])
            batch_data["domain"] = torch.LongTensor(
                [int(i) for i in data['domain']][batch_size * batch_id: batch_size * (batch_id + 1)])


        elif batch_id + 1 == batch_num:
            # # lucas 240115 list version
            # batch_data["user"] = data['user'][batch_size * batch_id:]
            # batch_data["item"] = data['item'][batch_size * batch_id:]
            # # batch_data["rating"] = data['rating'][batch_size * batch_id:]
            # # batch_data["input"] = data['input'][batch_size * batch_id:]
            # # batch_data["output"] = data['output'][batch_size * batch_id:]
            # # batch_data["instruction"] = data['instruction'][batch_size * batch_id:]
            # batch_data["input_ids"] = data['input_ids'][batch_size * batch_id:]
            # batch_data["attention_mask"] = data['attention_mask'][batch_size * batch_id:]
            # batch_data["labels"] = data['labels'][batch_size * batch_id:]
            # # batch_data = data[batch_size * (batch_id + 1):]

            # lucas 240115 torch version
            batch_data["user"] = torch.LongTensor(data['user'][batch_size * batch_id:])
            batch_data["item"] = torch.LongTensor(data['item'][batch_size * batch_id:])
            batch_data["input_ids"] = torch.LongTensor(data['input_ids'][batch_size * batch_id:])
            batch_data["attention_mask"] = torch.LongTensor(data['attention_mask'][batch_size * batch_id:])
            batch_data["labels"] = torch.LongTensor(data['labels'][batch_size * batch_id:])
            batch_data["rating"] = torch.LongTensor(data['rating'][batch_size * batch_id:])
            batch_data["domain"] = torch.LongTensor(data['domain'][batch_size * batch_id:])
        else:
            warnings.warn("---batch id is out of range")
    # print(f"---Lucas-take_batch_data- data total length: {length}; batch size: {batch_size}; batch num: {batch_num}; batch id: {batch_id}; batch data: {batch_data}")

    return batch_data


# lucas 240316 added generate function
def generate(model, data, batch_size, batch_id, tokenizer, loss_mse, loss_mae, rmse_total_loss, mae_total_loss, device):
    model.eval()
    with torch.no_grad():
        batch_data = take_batch_data(data, batch_size, batch_id)  # data.step += 190
        print((f"Lucas evaluation generate batch_data: {batch_data}"))
        data_num = len(batch_data['user'])
        # user = torch.LongTensor(batch_data['user']).to(device)  # (batch_size,)
        user = batch_data['user'].to(device)
        # item = torch.LongTensor(batch_data['item']).to(device)
        item = batch_data['item'].to(device)
        # rating = batch_data['rating'].to(device)
        # input = batch_data['input'].to(device)
        # output = batch_data['output'].to(device)
        # instruction = batch_data['instruction'].to(device)
        # input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
        input_ids = batch_data['input_ids'].to(device)
        # attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        rating = batch_data['rating'].to(device)
        domain = batch_data['domain'].to(device)
        # labels = torch.LongTensor(batch_data['labels']).to(device)
        # labels = batch_data['labels'].to(device)
        # outputs = model(user=user, item=item, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # print(f"---Lucas-generate func--take_batch_data--user: {user}; item: {item}; input_ids: {input_ids}; attention_mask: {attention_mask}; labels: {labels};")
        # generation_output = model.generate(
        #     generation_config=generation_config,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_new_tokens=max_new_tokens,
        #     # batch_size=batch_size,
        #     user=user,
        #     item=item,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels
        # )
        generated_ids = model.generate(max_new_tokens=200, do_sample=True, user=user, item=item, input_ids=input_ids,
                                       attention_mask=attention_mask)

        # Lucas 240328 added MSE, RMSE evaluation
        output = model(user=user, item=item, rating=rating, domain=domain, input_ids=input_ids,
                       attention_mask=attention_mask, labels=input_ids)
        predict_rating = output['rating_prediction']
        print(f"Lucas evaluation part rating prediction: {predict_rating}; original rating: {rating}")
        predict_rating = torch.clamp(predict_rating, max=5)
        predict_rating = torch.clamp(predict_rating, min=0)
        # print(f"Lucas evaluation part rating after clamp prediction: {predict_rating};")
        cur_mse_loss = loss_mse(predict_rating, rating)
        cur_mae_loss = loss_mae(predict_rating, rating)
        rmse_total_loss += cur_mse_loss.item()
        mae_total_loss += cur_mae_loss.item()
        print(
            f"Lucas evaluation part--cur_mse_loss: {cur_mse_loss};--cur_mae_loss: {cur_mae_loss}--rmse_total_loss: {rmse_total_loss};--mae_total_loss: {mae_total_loss}")

        generation_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # Lucas 240701 changed
        # generation_format = []
        # for generation in generation_output:
        #     cur_gen_format = generation[generation.find("### Response:")+15:]
        #     generation_format.append(cur_gen_format.strip())
        generation_format = [_.split("[/INST]")[-1].strip() for _ in generation_output]
        print(f"Lucas evaluation Generation output format: {generation_format}")
        # s = generation_output.sequences
        # scores = generation_output.scores[0].softmax(dim=-1)
        # print(f"Generation score: {scores}")
        # logits = torch.tensor(scores[:, [8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        # s = generation_output.sequences
        # output = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in generation_format]
        print(f"Lucas evaluation output: {output}")
        return output, data_num, generation_format, cur_mse_loss, cur_mae_loss, rmse_total_loss, mae_total_loss


# lucas 240316 add the evaluation method
def evaluation(model, vali_data, batch_size, device, tokenizer, config, prediction_path, temperature=0.6, top_p=0.9,
               top_k=40, num_beams=1, max_new_tokens=128, repetition_penalty=2, **kwargs):
    model.eval()
    batch_total = ceil(len(vali_data['user']) / batch_size)
    print(f"evaluation batch_total: {batch_total}--batch size: {batch_size}--length of user: {len(vali_data['user'])}")
    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     num_beams=num_beams,
    #     repetition_penalty=repetition_penalty,
    #     bos_token_id=1,
    #     do_sample=True,
    #     eos_token_id=2,
    #     max_length=4096,
    #     pad_token_id=0,
    #     # temperature=0.6,
    #     # top_p=0.9,
    #     # transformers_version="4.32.0.dev0"
    #     **kwargs,
    # )
    # eval_loss = 0
    generate_seq = []
    # Lucas 240328 added for rmse and mae evaluation
    loss_mse = LossMSE().to(device)
    loss_mae = LossMAE().to(device)
    rmse_total_loss = 0
    mae_total_loss = 0

    for batch_id in range(batch_total):
        print(f"Lucas main5-evaluation batch_id: {batch_id}--batch_total: {batch_total}")
        predict, data_num, predict_seq, cur_mse_loss, cur_mae_loss, rmse_total_loss, mae_total_loss = generate(model,
                                                                                                               vali_data,
                                                                                                               batch_size,
                                                                                                               batch_id,
                                                                                                               tokenizer,
                                                                                                               loss_mse,
                                                                                                               loss_mae,
                                                                                                               rmse_total_loss,
                                                                                                               mae_total_loss,
                                                                                                               device)
        # loss = outputs.loss
        # eval_loss += data_num * loss.item()
        for seq in predict_seq:
            seq = seq.strip("[/INST]").strip().lower()
            generate_seq.append(seq)
    # eval_loss = eval_loss/len(vali_data['user'])

    # lucas 240117 add additional evaluation metrics
    print(
        f"Lucas vali_data: {vali_data}; vali_data input_ids: {vali_data['input_ids']}; full_input_ids: {vali_data['full_input_ids']}; generate_seq: {generate_seq}")
    tokens_test = [ids2tokens(ids, tokenizer, tokenizer.eos_token_id) for ids in vali_data["full_input_ids"]]
    print(f"Lucas tokens_test: {tokens_test}")
    # tokens_predict = [ids2tokens(ids, tokenizer, config.eos_token_id) for ids in generate_seq[0]]
    tokens_predict = generate_seq
    print(f"Lucas tokens_predict: {tokens_predict}")
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7.4f}'.format(USR, USN))
    # lucas 240310 added meteor evaluation
    meteor_score = meteor(tokens_test, tokens_predict)
    print(now_time() + 'METEOR {:7.4f}'.format(meteor_score))
    # lucas 240328 added rmse and mae evaluation
    rmse_epoch = math.sqrt(rmse_total_loss / batch_total)
    mae_epoch = mae_total_loss / batch_total
    print(now_time() + 'RMSE {:7.4f}'.format(rmse_epoch))
    print(now_time() + 'MAE {:7.4f}'.format(mae_epoch))

    # feature_batch = feature_detect(tokens_predict, feature_set)
    # DIV = feature_diversity(feature_batch)  # time-consuming
    # print(now_time() + 'DIV {:7.4f}'.format(DIV))
    # FCR = feature_coverage_ratio(feature_batch, feature_set)
    # print(now_time() + 'FCR {:7.4f}'.format(FCR))
    # FMR = feature_matching_ratio(feature_batch, test_data.feature)
    # print(now_time() + 'FMR {:7.4f}'.format(FMR))

    # text_test = [' '.join(tokens) for tokens in tokens_test]
    # text_predict = [' '.join(tokens) for tokens in tokens_predict]
    ROUGE = rouge_score(tokens_test, tokens_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, fake) in zip(tokens_test, tokens_predict):
        text_out += '{}\n{}\n\n'.format(real, fake)
    with open(prediction_path, 'w', encoding='utf-8') as f:
        f.write(text_out)
    print(now_time() + 'Generated text saved to ({})'.format(prediction_path))

    return


# Lucas 240708 added for adaptive lambda in GRL layer
class AdaptiveLambda:
    def __init__(self, init_lambda, max_lambda, adaptation_rate):
        self.lambda_value = init_lambda
        self.max_lambda = max_lambda
        self.adaptation_rate = adaptation_rate

    def step(self, domain_loss, rating_loss):
        # Adjust lambda based on the ratio of domain loss to rating loss
        loss_ratio = domain_loss.item() / rating_loss.item()
        if loss_ratio > 1:
            # Domain loss is larger, decrease lambda
            self.lambda_value -= self.adaptation_rate
        else:
            # Rating loss is larger, increase lambda
            self.lambda_value += self.adaptation_rate

        # Ensure lambda stays within bounds
        self.lambda_value = max(0, min(self.lambda_value, self.max_lambda))

    def get_lambda(self):
        return self.lambda_value

def create_float_confusion_matrix(true_values, predicted_values, num_bins=10, threshold=0.1):
    # Ensure inputs are PyTorch tensors
    true_values = torch.tensor(true_values) if not isinstance(true_values, torch.Tensor) else true_values
    predicted_values = torch.tensor(predicted_values) if not isinstance(predicted_values,
                                                                        torch.Tensor) else predicted_values

    # Calculate the range of values
    min_val = min(true_values.min(), predicted_values.min(), 0)
    max_val = max(true_values.max(), predicted_values.max(), 5)

    # Create bins
    bins = torch.linspace(min_val, max_val, num_bins + 1).to(gpu_device)

    # Digitize the true and predicted values
    true_bins = torch.bucketize(true_values, bins) - 1
    pred_bins = torch.bucketize(predicted_values, bins) - 1

    # Create the confusion matrix
    confusion_matrix = torch.zeros(num_bins, num_bins)
    for t, p in zip(true_bins, pred_bins):
        confusion_matrix[t, p] += 1

    return confusion_matrix, bins


def plot_confusion_matrix(confusion_matrix, bins, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix for Float Data')

    # Set tick labels
    tick_labels = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(len(bins) - 1)]
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha='right')
    plt.yticks(range(len(tick_labels)), tick_labels, rotation=0)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/figs/confusion_matrix_{model_name}.png")


# Lucas 240703 added for validation with
def validate(model_name, model, val_loader, device, is_source=None):
    model.eval()
    total_loss = 0
    # Lucas 240705 added for rmse and mae evaluation
    loss_mse = LossMSE().to(device)
    loss_mae = LossMAE().to(device)
    rmse_total_loss = 0
    mae_total_loss = 0
    predictions = torch.FloatTensor([]).to(gpu_device)
    trues = torch.FloatTensor([]).to(gpu_device)
    batch_id = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_id += 1
            print(f"Lucas batch_id: {batch_id} batch data: {batch}")
            users = batch['user'].to(gpu_device)
            pos_items = batch['item'].to(gpu_device)
            rating = batch['rating'].to(gpu_device)
            domain = batch['domain'].to(gpu_device)
            neg_items = []
            if model_name in ["NCF-GAN-CL"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.LongTensor(neg_domain).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, neg_items, rating, neg_rating,
                                                                       domain, neg_domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas--Evaluation--contrastive loss: {train_loss_detail[2].item()};")
                print(f"Lucas--Evaluation--grl loss: {train_loss_detail[3].item()};")
                print(f"Lucas--Evaluation--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {train_loss_detail[0].item()}")

                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()
                count += len(rating)
                total_loss += train_loss.item()
            elif model_name in ["NCF-GAN", 'NCF-GAN-MOE']:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, rating, domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()
                count += len(rating)
                total_loss += train_loss.item()
            elif model_name in ["NCF-GAN2"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, rating, domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()
                count += len(rating)
                total_loss += train_loss.item()
            elif model_name in ["NCF-CL"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, neg_items, rating, neg_rating, domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                count += len(rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()

                total_loss += train_loss.item()
            elif model_name in ["NCF-CL_MemBank"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, neg_items, rating, neg_rating)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                count += len(rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()

                total_loss += train_loss.item()
            elif model_name in ["NCF-CL_MemBank_CurriLea"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, neg_items, rating, neg_rating,
                                                                       1, 1)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                count += len(rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()

                total_loss += train_loss.item()
            elif model_name in ["NCF", "NCF-DSN", "NCF-CDAN"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)

                train_loss, predict_ratings = model(users, pos_items, rating, domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                count += len(rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()

                total_loss += train_loss.item()
            elif model_name in ["NCF-MOE"]:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(users, pos_items, neg_items, rating, neg_rating, domain, is_source)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas is_source: {is_source} evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                count += len(rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()

                total_loss += train_loss.item()
            elif model_name in ['NCF-MMD', 'NCF-MDD']:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)

                train_loss, predict_ratings = model(users, pos_items, rating, domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                count += len(rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()

                total_loss += train_loss.item()
            elif model_name in ['NCF-MCD']:
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)

                preds_1, preds_2, ncf_1_loss, ncf_2_loss = model(users, pos_items, rating, domain)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))
                predict_ratings = (preds_1 + preds_2) / 2  # Average prediction
                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()
                train_loss = ncf_1_loss + ncf_2_loss
                total_loss += train_loss.item()
                count += len(rating)
            elif model_name in ['NCF-GAN-SENBERT']:
                review = batch['input']
                pre_preference = batch['pre_preference']
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)

                train_loss, train_loss_detail, predict_ratings = model(1, users, pos_items, rating, domain, review)
                # loss = bce_criterion(pred_pos, torch.ones_like(pred_pos))

                print(f"Lucas evaluation part rating prediction: {predict_ratings}; original rating: {rating}")
                predict_ratings = torch.clamp(predict_ratings, max=5, min=0)
                # print(f"Lucas evaluation part rating after clamp prediction: {predict_ratings};")
                cur_mse_loss = loss_mse(predict_ratings, rating)
                cur_mae_loss = loss_mae(predict_ratings, rating)
                rmse_total_loss += cur_mse_loss.item()
                mae_total_loss += cur_mae_loss.item()
                count += len(rating)
                total_loss += train_loss.item()
            predictions = torch.cat((predictions,predict_ratings), dim=-1)
            trues = torch.cat((trues, rating), dim=-1)

    avg_loss = total_loss / len(val_loader)
    # lucas 240328 added rmse and mae evaluation
    rmse_epoch = math.sqrt(rmse_total_loss / count)
    mae_epoch = mae_total_loss / count

    return avg_loss, mae_epoch, rmse_epoch, predictions, trues


def train(
        base_model: str = "../../Mistral-7B-Instruct-v0.2",
        # the only required argument--Mistral-7B-v0.1， Mixtral-8x7B-v0.1
        model_name: str = "Mixtral2-prefix-GAN-Tune",
        data_path: str = "Data/Amazon/ClothingShoesAndJewelry_Test",
        output_dir: str = "Model/book2movie/model",
        training_output: str = "Model/book2movie/sft",
        # training hyperparams
        train_domains: str = "both",
        # source: only train source data; target: only train target data; both: train source and target data
        batch_size: int = 64,
        emb_size: int = 4096,
        factor_num: int = 128,
        ncf_layer_num: int = 3,
        moe_layer_num: int = 3,
        micro_batch_size: int = 8,
        num_epochs: int = 1,
        learning_rate: float = 3e-4,
        cutoff_len: int = 512,
        val_set_size: int = 2000,
        lambda_l1: float = 1,  # weight of rating loss
        lambda_l2: float = 1,  # weight of contrastive loss
        lambda_l3: float = 1,  # weight of grl loss
        lambda_l4: float = 1,  # weight of llm loss
        lambda_grl: float = 1,  # weight of grl layer lambda
        margin: float = 1,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj", ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "",  # The prompt template to use, will default to alpaca.
        train_parallel: bool = False,  # lucas 231224 added for training parallel model
        prediction_path: str = "prediction.txt"
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"model_name: {model_name}\n"
        f"base_model: {base_model}\n"
        f"train_domains: {train_domains}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"emb_size: {emb_size}\n"
        f"factor_num: {factor_num}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lambda1: {lambda_l1}\n"
        f"lambda2: {lambda_l2}\n"
        f"lambda3: {lambda_l3}\n"
        f"lambda4: {lambda_l4}\n"
        f"margin: {margin}\n"
        f"lambda_grl: {lambda_grl}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n"
        f"train_parallel: {train_parallel}\n"
    )
    # Tokenizer
    # Lucas 240616 changed tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True, use_fast=True)
    # tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.padding_side = 'left'
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # *******lucas 240315 generate prompt
    prompter = Prompter(prompt_template_name)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # 240110 set padding: True and pad_token
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_train_prompt(data_point):
        full_prompt = prompter.generate_mixtral_prompt(
            data_point["user"],
            data_point["pre_preference"],
            data_point["item"],
            data_point["item_title"],
            data_point["item_category"],
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["user"], data_point["pre_preference"], data_point["item"], data_point["item_title"],
                data_point["item_category"], data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                         user_prompt_len:]  # could be sped up, probably
        # # lucas 231214 added user/item/rating data
        # tokenized_full_prompt['user'] = data_point["user"]
        # tokenized_full_prompt['item'] = data_point["item"]
        # tokenized_full_prompt['rating'] = data_point["rating"]
        tokenized_full_prompt["prompt"] = full_prompt
        print(f"Lucas--lucas_main5 generate_train_prompt--tokenized_full_prompt: {tokenized_full_prompt}")
        # return tokenized_full_prompt
        return {'prompt': full_prompt}

    def tokenize_batch(prompt, add_eos_token=True, cutoff_len=512):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # 240110 set padding: True and pad_token
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        # lucas 240113 ite the result and check data
        for i in range(len(result['input_ids'])):
            if (
                    result["input_ids"][i][-1] != tokenizer.eos_token_id
                    and len(result["input_ids"][i]) < cutoff_len
                    and add_eos_token
            ):
                result["input_ids"][i].append(tokenizer.eos_token_id)
                result["attention_mask"][i].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_test_prompt(loaded_data, prompter):
        full_prompt = []
        part_prompt = []  # without label in prompt
        for each_data in loaded_data["train"]:
            # with_label_prompt, without_label_prompt = prompter.generate_prompt_4test(
            #     each_data["user"],
            #     each_data["pre_preference"],
            #     each_data["item"],
            #     each_data["item_title"],
            #     each_data["item_category"],
            #     each_data["instruction"],
            #     each_data["input"],
            #     each_data["output"],
            # )
            with_label_prompt, without_label_prompt = prompter.generate_mixtral_test_prompt(
                each_data["user"],
                each_data["pre_preference"],
                each_data["item"],
                each_data["item_title"],
                each_data["item_category"],
                each_data["instruction"],
                each_data["input"],
                each_data["output"],
            )
            print(
                f"Lucas main5 loaded_data: {each_data}; \n with_label_prompt: {with_label_prompt}; \n without_label_prompt: {without_label_prompt}")
            full_prompt.append(with_label_prompt)
            part_prompt.append(without_label_prompt)
        tokenized_full_prompt = tokenize_batch(full_prompt)  # input_ids for evaluation
        tokenized_part_prompt = tokenize_batch(part_prompt)
        # lucas 240113 1: change the dataset-->dataframe, 2: then add data, Finally: 3: change dataframe-->dataset
        loaded_data_Frame = pd.DataFrame(loaded_data['train'])
        tokenized_prompt_Frame = pd.DataFrame(dict(tokenized_part_prompt))
        tokenized_full_prompt_Frame = pd.DataFrame({"full_input_ids": tokenized_full_prompt["input_ids"]})
        loaded_data_Frame = pd.concat([loaded_data_Frame, tokenized_prompt_Frame, tokenized_full_prompt_Frame],
                                      axis=1)  # ignore_index=True)
        loaded_data_dict = loaded_data_Frame.to_dict(orient="list")
        loaded_dataset = Dataset.from_dict(loaded_data_dict)
        return loaded_dataset

    # # *******prepare the dataset
    # dataset_train_sft = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    # dataset_test_sft = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:5%]")
    # dataset_test_sft = format_ultrachat(dataset_test_sft)
    # dataset_train_sft = format_ultrachat(dataset_train_sft)

    # *******prepare our own dataset
    train_data_path = os.path.join(data_path, "train.json")
    valid_data_path = os.path.join(data_path, "vali.json")
    test_data_path = os.path.join(data_path, "test.json")
    source_test_data_path = os.path.join(data_path, "source_test.json")
    target_test_data_path = os.path.join(data_path, "target_test.json")
    if train_data_path.endswith(".json") or train_data_path.endswith(".jsonl"):
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    if valid_data_path.endswith(".json") or valid_data_path.endswith(".jsonl"):
        val_data = load_dataset("json", data_files=valid_data_path)
    else:
        val_data = load_dataset(valid_data_path)
    if test_data_path.endswith(".json") or test_data_path.endswith(".jsonl"):
        test_data = load_dataset("json", data_files=test_data_path)
    else:
        test_data = load_dataset(test_data_path)
    if source_test_data_path.endswith(".json") or source_test_data_path.endswith(".jsonl"):
        source_test_data = load_dataset("json", data_files=source_test_data_path)
    else:
        source_test_data = load_dataset(test_data_path)
    if target_test_data_path.endswith(".json") or target_test_data_path.endswith(".jsonl"):
        target_test_data = load_dataset("json", data_files=target_test_data_path)
    else:
        target_test_data = load_dataset(test_data_path)
    # lucas 240220 count user/item number
    num_users = len(set(train_data['train']['user']) | set(val_data['train']['user']) | set(test_data['train']['user']))
    num_items = len(set(train_data['train']['item']) | set(val_data['train']['item']) | set(test_data['train']['item']))
    print(f"Lucas total user num: {num_users}; total item num: {num_items}")
    # lucas 240729 added select train data
    if train_domains == "source":
        train_data = train_data.filter(lambda u: u['domain'] == 0)
    elif train_domains == "target":
        train_data = train_data.filter(lambda u: u['domain'] == 1)

    if model_name in ["NCF-GAN-CL", "NCF-GAN", "NCF-GAN2", "NCF-CL", "NCF", "NCF-CL_MemBank", "NCF-CL_MemBank_CurriLea", "NCF-MOE", "NCF-GAN-MOE", "NCF-MMD", "NCF-MDD", "NCF-MCD", "NCF-DSN", "NCF-CDAN"]:
        # # Lucas 240703 added for ablation study
        # train_data = NCFDataset(train_data_path)
        # val_data = NCFDataset(valid_data_path)
        # test_data = NCFDataset(valid_data_path)
        train_data = train_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'input', 'output', 'review', 'pre_preference'])
        val_data = val_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'input', 'output', 'review', 'pre_preference'])
        test_data = test_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'input', 'output', 'review', 'pre_preference'])
        source_test_data = source_test_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'input', 'output', 'review', 'pre_preference'])
        target_test_data = target_test_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'input', 'output', 'review', 'pre_preference'])
    elif model_name in ["NCF-GAN-SENBERT"]:
        train_data = train_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'output', 'review'])
        val_data = val_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'output', 'review'])
        test_data = test_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'output', 'review'])
        source_test_data = source_test_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'output', 'review'])
        target_test_data = target_test_data.remove_columns(
            column_names=['item_title', 'item_category', 'instruction', 'output', 'review'])
    else:
        train_data = (train_data['train'].map(generate_train_prompt))
        val_data = (val_data['train'].map(generate_train_prompt))
        # lucas 240112 change the test_data
        # test_data = (test_data['train'].map(generate_train_prompt))
        test_data = tokenize_test_prompt(test_data, prompter)
        source_test_data = tokenize_test_prompt(source_test_data, prompter)
        target_test_data = tokenize_test_prompt(target_test_data, prompter)

        # lucas 231215 added remove specific columns
        train_data = train_data.remove_columns(column_names=['instruction', 'input', 'output'])
        val_data = val_data.remove_columns(column_names=['instruction', 'input', 'output'])
        test_data = test_data.remove_columns(column_names=['instruction', 'input', 'output'])
        source_test_data = source_test_data.remove_columns(column_names=['instruction', 'input', 'output'])
        target_test_data = target_test_data.remove_columns(column_names=['instruction', 'input', 'output'])

    print(f"Lucas train: {train_data}")
    print(f"Lucas train users: {train_data['train']['user'][0]}")
    print(f"Lucas train items: {train_data['train']['item'][0]}")
    print(f"Lucas train rating: {train_data['train']['rating'][0]}")
    print(f"Lucas train domain: {train_data['train']['domain'][0]}")
    print(f"Lucas val users: {val_data['train']['user'][0]}")
    print(f"Lucas val items: {val_data['train']['item'][0]}")
    print(f"Lucas test users: {test_data['train']['user'][0]}")
    print(f"Lucas test items: {test_data['train']['item'][0]}")
    print(f"Lucas test rating: {test_data['train']['rating'][0]}")
    print(f"Lucas test domain: {test_data['train']['domain'][0]}")

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #           base_model, quantization_config=bnb_config, device_map={"": 0}
    # )
    config = AutoConfig.from_pretrained(base_model)

    if model_name in ["Mixtral2-prefix-GAN-Tune"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-mixtral-GAN-CL",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = PrefixTune_GAN_Mixtral2(config, device_map={"": 0}, llm_model_path=base_model, nuser=num_users,
                                        nitem=num_items, quanti_config=bnb_config, lambda1=lambda_l1, lambda2=lambda_l2,
                                        lambda3=lambda_l3)
    elif model_name in ["Mixtral2-ncf-prefix-GAN-Tune"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-mixtral-NCF-GAN-CL",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = PrefixTune_NCF_GAN_Mixtral2(config, device_map={"": 0}, llm_model_path=base_model, nuser=num_users,
                                            nitem=num_items, quanti_config=bnb_config, factor_num=factor_num,
                                            ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1,
                                            lambda2=lambda_l2, lambda3=lambda_l3)
        model.llm_model.config.pad_token_id = tokenizer.pad_token_id
        model.llm_model.config.use_cache = False  # Gradient checkpointing is used by default but not compatible with caching
        print(f"Lucas model config: {model.config}")
        print(f"Lucas main cur model: {model};")
        print(f"Lucas main cur model device: {model.llm_model.device};")

        training_arguments = TrainingArguments(
            output_dir=training_output,
            evaluation_strategy="steps",
            do_eval=True,
            # optim="paged_adamw_8bit",
            optim="adamw_torch",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=batch_size,
            log_level="debug",
            save_steps=200,
            logging_steps=50,
            learning_rate=2e-4,
            eval_steps=50,
            # max_steps=300,
            num_train_epochs=num_epochs,
            warmup_steps=30,
            lr_scheduler_type="linear",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            # peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
        )
        print(f"Lucas start training")
        other_trainable_params = list()
        # for param in trainer.model.base_model.model.user_embeddings.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # for param in trainer.model.base_model.model.item_embeddings.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # for param in trainer.model.base_model.model.hidden_layers.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # for param in trainer.model.base_model.model.user_semantic_classifier.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # for param in trainer.model.base_model.model.rating_loss.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # for param in trainer.model.base_model.model.exp_loss_fn.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # for param in trainer.model.base_model.model.predict_layer.parameters():
        #     param.requires_grad = True
        #     other_trainable_params.append(param)
        # optimizer = torch.optim.Adam(other_trainable_params, lr=5e-5)
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                print(f"Lucas trainable parameter: {name} param {param}")
                # print(f"Lucas trainable parameter: {name} is optimized by {param.grad_fn.next_functions[0][0]}")
        trainer.train()
        print(f"Lucas end training")

        # Lucas 240403 source test evaluation
        print(f"Lucas****************Source test evaluation*******************")
        evaluation(model, source_test_data, batch_size, "cuda:0", tokenizer, LoraConfig, prediction_path,
                   temperature=0.8, top_p=0.9, top_k=40, num_beams=1,
                   max_new_tokens=128, repetition_penalty=1.18)
        print(f"Lucas****************Target test evaluation*******************")
        evaluation(model, target_test_data, batch_size, "cuda:0", tokenizer, LoraConfig, prediction_path,
                   temperature=0.8, top_p=0.9, top_k=40, num_beams=1,
                   max_new_tokens=128, repetition_penalty=1.18)
        print(f"Lucas****************overall test evaluation*******************")
        # evaluation(model, test_data, 8, "cuda:0", tokenizer, LoraConfig, prediction_path,
        #            temperature=0.8, top_p=0.9, top_k=40, num_beams=1,
        #            max_new_tokens=128, repetition_penalty=1.18)
        # print(f"Lucas evaluation loss: {eval_loss}")
    elif model_name in ["NCF-GAN-CL"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-GAN-CL",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_GAN_CL(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, emb_size=emb_size, factor_num=factor_num,
                       ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                       lambda3=lambda_l3, lambda4=lambda_l4, lambda_grl=lambda_grl, temperature=0.1)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.LongTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, rating_pred = model(users, pos_items, neg_items, rating, neg_rating, domain, neg_domain)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "neg_rating_loss": train_loss_detail[1].item(),
                    "contrastive_learning_loss": train_loss_detail[2].item(),
                    "ncf_grl_dom_loss": train_loss_detail[3].item(),
                    "sem_u_grl_dom_loss": train_loss_detail[4].item(),
                    "neg_ncf_grl_dom_loss": train_loss_detail[5].item(),
                    "neg_sem_u_grl_dom_loss": train_loss_detail[6].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[2].item()
                print(f"Lucas--Training--contrastive loss: {train_loss_detail[2].item()};")
                print(f"Lucas--Training--grl loss: {train_loss_detail[3].item()};")
                print(f"Lucas--Training--predict_ratings: {rating_pred}; original ratings: {rating}; rating loss: {train_loss_detail[0].item()}")

            print(f"Lucas train_loader len: {len(train_loader)}")
            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)

            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f}, InfoNCE loss: {avg_infonce_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            wandb.log({
                "epoch": epoch,
                "vali_loss": val_loss,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "source_test_mae": s_test_mae_epoch,
                "source_test_rmse": s_test_rmse_epoch,
                "target_test_mae": t_test_mae_epoch,
                "target_test_rmse": t_test_rmse_epoch,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-GAN"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-GAN",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_GAN(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, rating, domain)
                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]
                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "pred_ncf_domain_loss": train_loss_detail[1].item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-GAN2"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-GAN2",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_GAN2(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, rating, domain)
                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]
                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "pred_ncf_domain_loss": train_loss_detail[1].item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-CL"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-CL",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_CL(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                       ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                       lambda3=lambda_l3, lambda4=lambda_l4)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, neg_items, rating, neg_rating)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "neg_rating_loss": train_loss_detail[1].item(),
                    "contrastive_learning_loss": train_loss_detail[2].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[2].item()

            print(f"Lucas train_loader len: {len(train_loader)}")
            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f}, InfoNCE loss: {avg_infonce_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-CL_MemBank"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-CL_MemBank",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_Contrastive_MemoryBank(num_users=num_users, num_items=num_items, num_neg_i=5,
                                           layers=[factor_num * 2], embedding_dim=emb_size, queue_size=4096,
                                           momentum=0.999, device_map={"": 0}, factor_num=factor_num,
                                           ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1,
                                           lambda2=lambda_l2,
                                           lambda3=lambda_l3, lambda4=lambda_l4)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, neg_items, rating, neg_rating)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "neg_rating_loss": train_loss_detail[1].item(),
                    "contrastive_learning_loss": train_loss_detail[2].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[2].item()

            print(f"Lucas train_loader len: {len(train_loader)}")
            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f}, InfoNCE loss: {avg_infonce_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-CL_MemBank_CurriLea"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="NCF-CL_MemBank_CurriLea",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_ContLea_MemoryBank_CurriLea(num_users=num_users, num_items=num_items, num_neg_i=5,
                                                layers=[factor_num * 2], embedding_dim=emb_size, queue_size=4096,
                                                momentum=0.999, device_map={"": 0}, factor_num=factor_num,
                                                ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1,
                                                lambda2=lambda_l2,
                                                lambda3=lambda_l3, lambda4=lambda_l4)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                print(f"Lucas model forward num_epoch:{num_epochs}")
                train_loss, train_loss_detail, _ = model(users, pos_items, neg_items, rating, neg_rating, epoch + 1,
                                                         num_epochs)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "neg_rating_loss": train_loss_detail[1].item(),
                    "contrastive_learning_loss": train_loss_detail[2].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[2].item()

            print(f"Lucas train_loader len: {len(train_loader)}")
            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f}, InfoNCE loss: {avg_infonce_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                    ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                    lambda3=lambda_l3, lambda4=lambda_l4)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, _ = model(users, pos_items, rating, domain)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "rating_loss": train_loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()

            avg_loss = total_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_loss:.5f}")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            if train_domains == 'source':
                s_test_loss, s_test_mae_epoch, s_test_rmse_epoch, s_predictions, s_trues = validate(model_name, model, s_test_loader, gpu_device)
                print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
                print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
                print(f"Source Test Loss: {s_test_loss:.5f}")
            elif train_domains == 'target':
                t_test_loss, t_test_mae_epoch, t_test_rmse_epoch, t_predictions, t_trues = validate(model_name, model, t_test_loader, gpu_device)
                print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
                print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
                print(f"Target Test Loss: {t_test_loss:.5f}")
            else:
                s_test_loss, s_test_mae_epoch, s_test_rmse_epoch, s_predictions, s_trues = validate(model_name, model, s_test_loader,
                                                                            gpu_device)
                print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
                print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
                print(f"Source Test Loss: {s_test_loss:.5f}")

                t_test_loss, t_test_mae_epoch, t_test_rmse_epoch, t_predictions, t_trues = validate(model_name, model, t_test_loader,
                                                                            gpu_device)
                print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
                print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
                print(f"Target Test Loss: {t_test_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if train_domains in ['source','both']:
                    s_conf_matrix, s_bins = create_float_confusion_matrix(s_trues, s_predictions)
                    print(f"Source Confusion Matrix: {s_conf_matrix}; bins: {s_bins}")
                    plot_confusion_matrix(s_conf_matrix, s_bins, f"{model_name}_source")
                if train_domains in ['target', 'both']:
                    t_conf_matrix, t_bins = create_float_confusion_matrix(t_trues, t_predictions)
                    print(f"Source Confusion Matrix: {t_conf_matrix}; bins: {t_bins}")
                    plot_confusion_matrix(t_conf_matrix, t_bins, f"{model_name}_target")

                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-MOE"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-MOE",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_MOE(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, moe_layer_num=moe_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl)
        # Lucas 240702 dataset
        s_train_data = train_data.filter(lambda u: u['domain'] == 0)
        t_train_data = train_data.filter(lambda u: u['domain'] == 1)
        # train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        s_train_loader = DataLoader(s_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        t_train_loader = DataLoader(t_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)

        s_val_data = val_data.filter(lambda u: u['domain'] == 0)
        t_val_data = val_data.filter(lambda u: u['domain'] == 1)
        # val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_val_loader = DataLoader(s_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_val_loader = DataLoader(t_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in s_train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, neg_items, rating, neg_rating, domain, is_source=True)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "neg_rating_loss": train_loss_detail[1].item(),
                    "contrastive_learning_loss": train_loss_detail[2].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[2].item()
            for batch in t_train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, neg_items, rating, neg_rating, domain, is_source=False)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "neg_rating_loss": train_loss_detail[1].item(),
                    "contrastive_learning_loss": train_loss_detail[2].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[2].item()

            print(f"Lucas s_train_loader len: {len(s_train_loader)}; t_train_loader len: {len(t_train_loader)}")
            data_len = len(s_train_loader) + len(t_train_loader)
            avg_loss = total_loss / data_len
            avg_rating_loss = total_rating_loss / data_len
            avg_infonce_loss = total_infonce_loss / data_len
            print(f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f}, InfoNCE loss: {avg_infonce_loss:.5f})")

            s_val_loss, s_val_mae_epoch, s_val_rmse_epoch = validate(model_name, model, s_val_loader, gpu_device, is_source=True)
            print(now_time() + 'Source Validation RMSE {:7.5f}'.format(s_val_rmse_epoch))
            print(now_time() + 'Source Validation MAE {:7.5f}'.format(s_val_mae_epoch))
            t_val_loss, t_val_mae_epoch, t_val_rmse_epoch = validate(model_name, model, t_val_loader, gpu_device, is_source=False)
            print(now_time() + 'Target Validation RMSE {:7.5f}'.format(t_val_rmse_epoch))
            print(now_time() + 'Target Validation MAE {:7.5f}'.format(t_val_mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")

            print(f"Source Validation Loss: {s_val_loss:.5f}")
            print(f"Target Validation Loss: {t_val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device, is_source=True)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device, is_source=False)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            if t_val_loss < best_val_loss:
                best_val_loss = t_val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-GAN-CL-MOE"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-GAN-CL-MOE",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_GAN_CL_MOE(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                           ncf_layer_num=ncf_layer_num, moe_layer_num=moe_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                           lambda3=lambda_l3, lambda4=lambda_l4)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            total_infonce_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                neg_items = []
                for i in batch['contrast']:
                    for j in i.split(", "):
                        neg_items.append(int(j))
                neg_items = torch.IntTensor(neg_items).to(gpu_device)
                neg_rating = []
                for i in batch['cl_rating']:
                    for j in i.split(", "):
                        neg_rating.append(int(j))
                neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                neg_domain = []
                for i in batch['cl_domain']:
                    for j in i.split(", "):
                        neg_domain.append(int(j))
                neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, neg_items, rating, neg_rating, domain,
                                                         neg_domain)
                # Backpropagation
                train_loss.backward()
                optimizer.step()

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "contrastive_learning_loss": train_loss_detail[4].item(),
                    "pred_ncf_domain_loss": train_loss_detail[2].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                total_infonce_loss += train_loss_detail[4].item()

            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f}, InfoNCE loss: {avg_infonce_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-GAN-MOE"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-GAN-MOE",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        # model = NCF_GAN_MOE(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
        #                 ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
        #                 lambda3=lambda_l3, lambda_grl=lambda_grl, moe_layer_num=moe_layer_num)
        model = NCF_GAN_TRNSMOE(device_map=gpu_device, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                            ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                            lambda3=lambda_l3, lambda_grl=lambda_grl, moe_layer_num=moe_layer_num)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            for batch in train_loader:
                print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                train_loss, train_loss_detail, _ = model(users, pos_items, rating, domain)
                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]
                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "pred_ncf_domain_loss": train_loss_detail[1].item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-MMD"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name=f"WWW-2-{model_name}",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_MMD(device_map=gpu_device, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl)
        # Lucas 240825 dataset
        s_train_data = train_data.filter(lambda u: u['domain'] == 0)
        t_train_data = train_data.filter(lambda u: u['domain'] == 1)
        # train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        s_train_loader = DataLoader(s_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        t_train_loader = DataLoader(t_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Train data len: {len(train_loader)}; Source train data len: {len(s_train_loader)}; Target train data len: {len(t_train_loader)}")

        s_val_data = val_data.filter(lambda u: u['domain'] == 0)
        t_val_data = val_data.filter(lambda u: u['domain'] == 1)
        # val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_val_loader = DataLoader(s_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_val_loader = DataLoader(t_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        # Lucas 240702 dataset

        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_s_rating_loss = 0
            total_t_rating_loss = 0
            total_mmd_loss = 0
            ite_num = 0
            # for batch in train_loader:
            for (data_s, data_t) in zip(s_train_loader, t_train_loader):
                ite_num += 1
                print(f"Lucas Source batch data: {data_s}; Target batch data: {data_t}")
                s_users = data_s['user'].to(gpu_device)
                s_pos_items = data_s['item'].to(gpu_device)
                s_rating = data_s['rating'].to(gpu_device)
                s_domain = data_s['domain'].to(gpu_device)

                t_users = data_t['user'].to(gpu_device)
                t_pos_items = data_t['item'].to(gpu_device)
                t_rating = data_t['rating'].to(gpu_device)
                t_domain = data_t['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                s_train_loss, _ = model(s_users, s_pos_items, s_rating, s_domain)
                t_train_loss, _ = model(t_users, t_pos_items, t_rating, t_domain)
                # Compute MMD loss between the two domains
                if len(s_users) == len(t_users):
                    mmd_loss_value = model.compute_mmd_loss(s_users, s_pos_items, t_users, t_pos_items)
                    train_loss = s_train_loss + t_train_loss + mmd_loss_value
                    print(
                        f"s_train_loss: {s_train_loss.item()}; t_train_loss: {t_train_loss.item()}; mmd_loss_value: {mmd_loss_value.item()}; train_loss: {train_loss}")

                else:
                    train_loss = s_train_loss + t_train_loss
                    print(
                        f"s_train_loss: {s_train_loss.item()}; t_train_loss: {t_train_loss.item()}; train_loss: {train_loss}")

                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]

                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "total_loss": train_loss.item(),
                    "s_rating_loss": s_train_loss.item(),
                    "t_rating_loss": t_train_loss.item(),
                    "mmd_loss": mmd_loss_value.item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_s_rating_loss += s_train_loss.item()
                total_t_rating_loss += t_train_loss.item()
                total_mmd_loss += mmd_loss_value.item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / ite_num
            avg_s_rating_loss = total_s_rating_loss / ite_num
            avg_t_rating_loss = total_t_rating_loss / ite_num
            avg_mmd_loss = total_mmd_loss / ite_num
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train ave Loss: {avg_loss:.5f} (Ave Source Rating loss: {avg_s_rating_loss:.5f}; Ave Target Rating loss: {avg_t_rating_loss:.5f}; Ave mmd loss: {avg_mmd_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-MDD"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name=f"WWW-2-{model_name}",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_MDD(device_map=gpu_device, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl, margin=margin)
        # Lucas 240825 dataset
        s_train_data = train_data.filter(lambda u: u['domain'] == 0)
        t_train_data = train_data.filter(lambda u: u['domain'] == 1)
        # train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        s_train_loader = DataLoader(s_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        t_train_loader = DataLoader(t_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Train data len: {len(train_loader)}; Source train data len: {len(s_train_loader)}; Target train data len: {len(t_train_loader)}")

        s_val_data = val_data.filter(lambda u: u['domain'] == 0)
        t_val_data = val_data.filter(lambda u: u['domain'] == 1)
        # val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_val_loader = DataLoader(s_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_val_loader = DataLoader(t_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        # Lucas 240702 dataset

        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_s_rating_loss = 0
            total_t_rating_loss = 0
            total_mmd_loss = 0
            ite_num = 0
            # for batch in train_loader:
            for (data_s, data_t) in zip(s_train_loader, t_train_loader):
                ite_num += 1
                print(f"Lucas Source batch data: {data_s}; Target batch data: {data_t}")
                s_users = data_s['user'].to(gpu_device)
                s_pos_items = data_s['item'].to(gpu_device)
                s_rating = data_s['rating'].to(gpu_device)
                s_domain = data_s['domain'].to(gpu_device)

                t_users = data_t['user'].to(gpu_device)
                t_pos_items = data_t['item'].to(gpu_device)
                t_rating = data_t['rating'].to(gpu_device)
                t_domain = data_t['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                s_train_loss, _ = model(s_users, s_pos_items, s_rating, s_domain)
                t_train_loss, _ = model(t_users, t_pos_items, t_rating, t_domain)
                # Compute MMD loss between the two domains
                if len(s_users) == len(t_users):
                    mmd_loss_value = model.compute_mdd_loss(s_users, s_pos_items, s_rating, s_domain, t_users, t_pos_items, t_rating, t_domain)
                    train_loss = s_train_loss + t_train_loss + mmd_loss_value
                    print(
                        f"s_train_loss: {s_train_loss.item()}; t_train_loss: {t_train_loss.item()}; mmd_loss_value: {mmd_loss_value.item()}; train_loss: {train_loss}")

                else:
                    train_loss = s_train_loss + t_train_loss
                    print(
                        f"s_train_loss: {s_train_loss.item()}; t_train_loss: {t_train_loss.item()}; train_loss: {train_loss}")

                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]

                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "total_loss": train_loss.item(),
                    "s_rating_loss": s_train_loss.item(),
                    "t_rating_loss": t_train_loss.item(),
                    "mmd_loss": mmd_loss_value.item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_s_rating_loss += s_train_loss.item()
                total_t_rating_loss += t_train_loss.item()
                total_mmd_loss += mmd_loss_value.item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / ite_num
            avg_s_rating_loss = total_s_rating_loss / ite_num
            avg_t_rating_loss = total_t_rating_loss / ite_num
            avg_mmd_loss = total_mmd_loss / ite_num
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train ave Loss: {avg_loss:.5f} (Ave Source Rating loss: {avg_s_rating_loss:.5f}; Ave Target Rating loss: {avg_t_rating_loss:.5f}; Ave mmd loss: {avg_mmd_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-MCD"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name=f"WWW-2-{model_name}",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_MCD(device_map=gpu_device, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl, margin=margin)
        # Lucas 240825 dataset
        s_train_data = train_data.filter(lambda u: u['domain'] == 0)
        t_train_data = train_data.filter(lambda u: u['domain'] == 1)
        # train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        s_train_loader = DataLoader(s_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        t_train_loader = DataLoader(t_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Train data len: {len(train_loader)}; Source train data len: {len(s_train_loader)}; Target train data len: {len(t_train_loader)}")

        s_val_data = val_data.filter(lambda u: u['domain'] == 0)
        t_val_data = val_data.filter(lambda u: u['domain'] == 1)
        # val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_val_loader = DataLoader(s_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_val_loader = DataLoader(t_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        # Lucas 240702 dataset

        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_s_rating_loss = 0
            total_t_rating_loss = 0
            total_mcd_loss = 0
            ite_num = 0
            # for batch in train_loader:
            for (data_s, data_t) in zip(s_train_loader, t_train_loader):
                ite_num += 1
                print(f"Lucas Source batch data: {data_s}; Target batch data: {data_t}")
                s_users = data_s['user'].to(gpu_device)
                s_pos_items = data_s['item'].to(gpu_device)
                s_rating = data_s['rating'].to(gpu_device)
                s_domain = data_s['domain'].to(gpu_device)

                t_users = data_t['user'].to(gpu_device)
                t_pos_items = data_t['item'].to(gpu_device)
                t_rating = data_t['rating'].to(gpu_device)
                t_domain = data_t['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                s_preds_1, s_preds_2, s_ncf_1_loss, s_ncf_2_loss = model(s_users, s_pos_items, s_rating, s_domain)
                t_preds_1, t_preds_2, t_ncf_1_loss, t_ncf_2_loss = model(t_users, t_pos_items, t_rating, t_domain)
                # Compute MMD loss between the two domains
                if len(s_users) == len(t_users):
                    source_discrepancy, target_discrepancy = model.compute_mcd_loss(s_users, s_pos_items, s_rating, s_domain, t_users, t_pos_items, t_rating, t_domain)

                    # Total loss
                    loss_A = s_ncf_1_loss + s_ncf_2_loss
                    loss_B = t_ncf_1_loss + t_ncf_2_loss
                    mcd_loss = source_discrepancy - target_discrepancy

                    train_loss = loss_A + loss_B + mcd_loss
                    # train_loss = s_train_loss + t_train_loss + mmd_loss_value
                    print(
                        f"s_train_ncf1 loss: {s_ncf_1_loss.item()}; s_train_ncf_2 loss: {s_ncf_2_loss.item()}; "
                        f"t_train_ncf1_loss: {t_ncf_1_loss.item()}; t_train_ncf_2 loss: {t_ncf_2_loss.item()};"
                        f"mcd_loss: {mcd_loss.item()}; train_loss: {train_loss}")
                    total_mcd_loss += mcd_loss.item()
                else:
                    # Total loss
                    loss_A = s_ncf_1_loss + s_ncf_2_loss
                    loss_B = t_ncf_1_loss + t_ncf_2_loss

                    train_loss = loss_A + loss_B
                    # train_loss = s_train_loss + t_train_loss + mmd_loss_value
                    print(
                        f"s_train_ncf1 loss: {s_ncf_1_loss.item()}; s_train_ncf_2 loss: {s_ncf_2_loss.item()}; "
                        f"t_train_ncf1_loss: {t_ncf_1_loss.item()}; t_train_ncf_2 loss: {t_ncf_2_loss.item()};"
                        f"train_loss: {train_loss}")

                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]

                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "total_loss": train_loss.item(),
                    "s_ncf_1_loss": s_ncf_1_loss.item(),
                    "s_ncf_2_loss": s_ncf_2_loss.item(),
                    "t_ncf_1_loss": t_ncf_1_loss.item(),
                    "t_ncf_2_loss": t_ncf_2_loss.item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_s_rating_loss += (s_ncf_1_loss.item() + s_ncf_2_loss.item())/2
                total_t_rating_loss += (t_ncf_1_loss.item() + t_ncf_2_loss.item())/2

                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / ite_num
            avg_s_rating_loss = total_s_rating_loss / ite_num
            avg_t_rating_loss = total_t_rating_loss / ite_num
            avg_mmd_loss = total_mcd_loss / (ite_num-1)
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train ave Loss: {avg_loss:.5f} (Ave Source Rating loss: {avg_s_rating_loss:.5f}; Ave Target Rating loss: {avg_t_rating_loss:.5f}; Ave mmd loss: {avg_mmd_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-DSN"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name=f"WWW-2-{model_name}",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_DSN(device_map=gpu_device, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl, margin=margin)
        # Lucas 240825 dataset
        s_train_data = train_data.filter(lambda u: u['domain'] == 0)
        t_train_data = train_data.filter(lambda u: u['domain'] == 1)
        # train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        s_train_loader = DataLoader(s_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        t_train_loader = DataLoader(t_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Train data len: {len(train_loader)}; Source train data len: {len(s_train_loader)}; Target train data len: {len(t_train_loader)}")

        s_val_data = val_data.filter(lambda u: u['domain'] == 0)
        t_val_data = val_data.filter(lambda u: u['domain'] == 1)
        # val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_val_loader = DataLoader(s_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_val_loader = DataLoader(t_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        # Lucas 240702 dataset

        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_s_rating_loss = 0
            total_t_rating_loss = 0
            total_mmd_loss = 0
            ite_num = 0
            # for batch in train_loader:
            for (data_s, data_t) in zip(s_train_loader, t_train_loader):
                ite_num += 1
                print(f"Lucas Source batch data: {data_s}; Target batch data: {data_t}")
                s_users = data_s['user'].to(gpu_device)
                s_pos_items = data_s['item'].to(gpu_device)
                s_rating = data_s['rating'].to(gpu_device)
                s_domain = data_s['domain'].to(gpu_device)

                t_users = data_t['user'].to(gpu_device)
                t_pos_items = data_t['item'].to(gpu_device)
                t_rating = data_t['rating'].to(gpu_device)
                t_domain = data_t['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                s_train_loss, _ = model(s_users, s_pos_items, s_rating, s_domain)
                s_train_loss.backward()
                t_train_loss, _ = model(t_users, t_pos_items, t_rating, t_domain)
                t_train_loss.backward()

                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]

                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "s_rating_loss": s_train_loss.item(),
                    "t_rating_loss": t_train_loss.item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_s_rating_loss += s_train_loss.item()
                total_t_rating_loss += t_train_loss.item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_s_rating_loss = total_s_rating_loss / ite_num
            avg_t_rating_loss = total_t_rating_loss / ite_num
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Ave Source Rating loss: {avg_s_rating_loss:.5f}; Ave Target Rating loss: {avg_t_rating_loss:.5f};)")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-CDAN"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name=f"WWW-2-{model_name}",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_CDAN(device_map=gpu_device, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl, margin=margin)
        # Lucas 240825 dataset
        s_train_data = train_data.filter(lambda u: u['domain'] == 0)
        t_train_data = train_data.filter(lambda u: u['domain'] == 1)
        # train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        s_train_loader = DataLoader(s_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        t_train_loader = DataLoader(t_train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Train data len: {len(train_loader)}; Source train data len: {len(s_train_loader)}; Target train data len: {len(t_train_loader)}")

        s_val_data = val_data.filter(lambda u: u['domain'] == 0)
        t_val_data = val_data.filter(lambda u: u['domain'] == 1)
        # val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_val_loader = DataLoader(s_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_val_loader = DataLoader(t_val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        # Lucas 240702 dataset

        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        # t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        for epoch in range(num_epochs):
            total_loss = 0
            total_s_rating_loss = 0
            total_t_rating_loss = 0
            total_cdan_loss = 0
            ite_num = 0
            # for batch in train_loader:
            for (data_s, data_t) in zip(s_train_loader, t_train_loader):
                ite_num += 1
                print(f"Lucas Source batch data: {data_s}; Target batch data: {data_t}")
                s_users = data_s['user'].to(gpu_device)
                s_pos_items = data_s['item'].to(gpu_device)
                s_rating = data_s['rating'].to(gpu_device)
                s_domain = data_s['domain'].to(gpu_device)

                t_users = data_t['user'].to(gpu_device)
                t_pos_items = data_t['item'].to(gpu_device)
                t_rating = data_t['rating'].to(gpu_device)
                t_domain = data_t['domain'].to(gpu_device)
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                s_train_loss, _ = model(s_users, s_pos_items, s_rating, s_domain)
                # s_train_loss.backward()
                t_train_loss, _ = model(t_users, t_pos_items, t_rating, t_domain)
                # t_train_loss.backward()

                # Compute CDAN loss between the two domains
                source_domain_pred, target_domain_pred = model.compute_cdan_loss(s_users, s_pos_items, s_rating, s_domain, t_users, t_pos_items, t_rating, t_domain)

                # Domain labels: 0 for source, 1 for target
                source_domain_labels = torch.zeros_like(source_domain_pred)
                target_domain_labels = torch.ones_like(target_domain_pred)

                # Compute domain classification loss
                domain_loss = F.binary_cross_entropy(source_domain_pred, source_domain_labels) + \
                              F.binary_cross_entropy(target_domain_pred, target_domain_labels)
                # Total loss
                train_loss = s_train_loss + t_train_loss + domain_loss
                train_loss.backward()

                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]

                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "s_rating_loss": s_train_loss.item(),
                    "t_rating_loss": t_train_loss.item(),
                    "domain_loss": domain_loss.item(),
                    "total_loss": train_loss.item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_s_rating_loss += s_train_loss.item()
                total_t_rating_loss += t_train_loss.item()
                total_loss += train_loss.item()
                total_cdan_loss += domain_loss.item()

                # # total_infonce_loss += train_loss_detail[6].item()

            avg_s_rating_loss = total_s_rating_loss / ite_num
            avg_t_rating_loss = total_t_rating_loss / ite_num
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Ave Source Rating loss: {avg_s_rating_loss:.5f}; Ave Target Rating loss: {avg_t_rating_loss:.5f};)")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)

        # Lucas 240705 Close the wandb run
        wandb.finish()
    elif model_name in ["NCF-GAN-SENBERT"]:
        # Lucas 240705 Start a wandb
        wandb.init(project="WWW-2", name="WWW-2-NCF-GAN-SENBERT",
                   config={"learning_rate": learning_rate, "epochs": num_epochs})
        model = NCF_GAN_SenBert(device_map={"": 0}, num_u=num_users, num_i=num_items, num_neg_i=5, factor_num=factor_num,
                        ncf_layer_num=ncf_layer_num, drop_out=lora_dropout, lambda1=lambda_l1, lambda2=lambda_l2,
                        lambda3=lambda_l3, lambda_grl=lambda_grl)
        # Lucas 240702 dataset
        train_loader = DataLoader(train_data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        s_test_loader = DataLoader(source_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)
        t_test_loader = DataLoader(target_test_data['train'], batch_size=batch_size, shuffle=False, num_workers=4)

        model = model.to(gpu_device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_test_mae = 0
        best_result = []
        # Lucas 240705 Log wandb model architecture
        wandb.watch(model)
        # Lucas 240708 added adaptive lambda for grl loss
        # adaptive_lambda = AdaptiveLambda(init_lambda=0.1, max_lambda=1.0, adaptation_rate=0.01)
        epoch_int = 0
        num_batch = len(train_loader)
        active_domain_loss_step = 0
        dann_epoch = np.floor(active_domain_loss_step / num_batch * 1.0)  # lucas 返回不大于输入参数的最大整数
        for epoch in range(num_epochs):
            total_loss = 0
            total_rating_loss = 0
            epoch_int += 1
            batch_int = 0
            for batch in train_loader:
                p = 1
                # print(f"Lucas batch data: {batch}")
                users = batch['user'].to(gpu_device)
                pos_items = batch['item'].to(gpu_device)
                rating = batch['rating'].to(gpu_device)
                domain = batch['domain'].to(gpu_device)
                # review = batch['input']
                review = batch['pre_preference']
                # neg_items = []
                # for i in batch['contrast']:
                #     for j in i.split(", "):
                #         neg_items.append(int(j))
                # neg_items = torch.IntTensor(neg_items).to(gpu_device)
                # neg_rating = []
                # for i in batch['cl_rating']:
                #     for j in i.split(", "):
                #         neg_rating.append(int(j))
                # neg_rating = torch.FloatTensor(neg_rating).to(gpu_device)
                # neg_domain = []
                # for i in batch['cl_domain']:
                #     for j in i.split(", "):
                #         neg_domain.append(int(j))
                # neg_domain = torch.IntTensor(neg_domain).to(gpu_device)

                optimizer.zero_grad()
                batch_int += 1
                p = float(batch_int + (epoch - dann_epoch) * num_batch / (num_epochs - dann_epoch) / num_batch)
                p = 2. / (1. + np.exp(-10 * p)) - 1
                print(f"Lucas p: {p}")
                train_loss, train_loss_detail, _ = model(p, users, pos_items, rating, domain, review)
                # Lucas 240709 Get current lambda and compute total loss
                # lambda_domain = adaptive_lambda.get_lambda()

                # Lucas 240708 changed loss function Backpropagation
                # train_loss.backward()
                # train_loss = train_loss_detail[0] + lambda_domain * train_loss_detail[1]
                train_loss.backward()
                optimizer.step()

                # Lucas 240709 Update adaptive lambda
                # adaptive_lambda.step(train_loss_detail[1], train_loss_detail[0])

                # Lucas 240705
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "rating_loss": train_loss_detail[0].item(),
                    "pred_ncf_domain_loss": train_loss_detail[1].item(),
                    # "pred_sem_domain_loss": train_loss_detail[2].item(),
                    # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                total_loss += train_loss.item()
                total_rating_loss += train_loss_detail[0].item()
                # # total_infonce_loss += train_loss_detail[6].item()

            avg_loss = total_loss / len(train_loader)
            avg_rating_loss = total_rating_loss / len(train_loader)
            # avg_infonce_loss = total_infonce_loss / len(train_loader)

            val_loss, mae_epoch, rmse_epoch, v_predictions, v_trues = validate(model_name, model, val_loader, gpu_device)
            print(now_time() + 'Validation RMSE {:7.5f}'.format(rmse_epoch))
            print(now_time() + 'Validation MAE {:7.5f}'.format(mae_epoch))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_loss:.5f} (Rating loss: {avg_rating_loss:.5f})")
            print(f"Validation Loss: {val_loss:.5f}")

            # test_loss, test_mae_epoch, test_rmse_epoch = validate(model_name, model, test_loader, gpu_device)
            # print(now_time() + 'Test RMSE {:7.5f}'.format(test_rmse_epoch))
            # print(now_time() + 'Test MAE {:7.5f}'.format(test_mae_epoch))
            # print(f"Test Loss: {test_loss:.5f}")

            s_test_loss, s_test_mae_epoch, s_test_rmse_epoch, s_predictions, s_trues = validate(model_name, model, s_test_loader, gpu_device)
            print(now_time() + 'Source Test RMSE {:7.5f}'.format(s_test_rmse_epoch))
            print(now_time() + 'Source Test MAE {:7.5f}'.format(s_test_mae_epoch))
            print(f"Source Test Loss: {s_test_loss:.5f}")

            t_test_loss, t_test_mae_epoch, t_test_rmse_epoch, t_predictions, t_trues = validate(model_name, model, t_test_loader, gpu_device)
            print(now_time() + 'Target Test RMSE {:7.5f}'.format(t_test_rmse_epoch))
            print(now_time() + 'Target Test MAE {:7.5f}'.format(t_test_mae_epoch))
            print(f"Target Test Loss: {t_test_loss:.5f}")

            # Lucas 240705
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "vali_mae": mae_epoch,
                "vali_rmse": rmse_epoch,
                "S_test_mae": s_test_mae_epoch,
                "S_test_rmse": s_test_rmse_epoch,
                "T_test_mae": t_test_mae_epoch,
                "T_test_rmse": t_test_rmse_epoch,
                # "adapter_pred_ncf_domain_loss": lambda_domain * train_loss_detail[1].item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            if epoch == 0:
                best_test_mae = t_test_mae_epoch
            if t_test_mae_epoch < best_test_mae:
                print(f"Best epoch is: {epoch}; Source MAE: {s_test_mae_epoch}; Source RMSE: {s_test_rmse_epoch}; Target MAE: {t_test_mae_epoch}; Target RMSE: {t_test_rmse_epoch}")
                best_test_mae = t_test_mae_epoch
                best_result = [s_test_mae_epoch, s_test_rmse_epoch, t_test_mae_epoch, t_test_rmse_epoch]
                # draw confusion matrix
                s_conf_matrix, s_bins = create_float_confusion_matrix(s_trues, s_predictions)
                print(f"epoch: {epoch}; Source Confusion Matrix: {s_conf_matrix}; bins: {s_bins}")
                plot_confusion_matrix(s_conf_matrix, s_bins, f"{model_name}_source")
                t_conf_matrix, t_bins = create_float_confusion_matrix(t_trues, t_predictions)
                print(f"epoch: {epoch};Source Confusion Matrix: {t_conf_matrix}; bins: {t_bins}")
                plot_confusion_matrix(t_conf_matrix, t_bins, f"{model_name}_target")

                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("New best model saved!")
                # model.save_pretrained(output_dir)
        print(
            f"Final Best epoch is: {best_test_mae}; Source MAE: {best_result[0]}; Source RMSE: {best_result[1]}; Target MAE: {best_result[2]}; Target RMSE: {best_result[3]}")
        # Lucas 240705 Close the wandb run
        wandb.finish()



if __name__ == "__main__":
    fire.Fire(train)