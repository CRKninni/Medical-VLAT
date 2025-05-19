import os
import yaml
import wandb 
import utils
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, RobertaTokenizer


from models.MAIN_VLAT import VLAT, VLATClipRoberta
from models.vit import interpolate_pos_embed

from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from optim import create_optimizer
from scheduler import create_scheduler


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps * step_size  

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations: 
            scheduler.step(i // step_size) 

        # Log to wandb at every print frequency (or adjust as needed)
        if i % print_freq == 0:
            wandb.log({
                "train/loss": float(loss.item()),  # Explicitly cast to float
                "train/lr": float(optimizer.param_groups[0]["lr"]),  # Ensure lr is float
                "epoch": int(epoch),  # Ensure epoch is integer
                "step": int(i)  # Ensure step is integer
            })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, None, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})   

    return result

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def main(args, config):
    # Initialize wandb for experiment tracking
    wandb.init(project="VQA_SLAKE", config=config)
    
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 1
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(
        datasets, samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test']],
        num_workers=[4, 4], is_trains=[True, False], 
        collate_fns=[vqa_collate_fn, None]
    ) 
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    # tokenizer = RobertaTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = VLAT(config_file="configs/config_bert.json", decoder_base=args.text_decoder)
    # model = VLATClipRoberta()
    model = model.to(device)   
    total_params = count_parameters(model)
    # Log to wandb
    wandb.log({"Total Parameters": total_params})   
    print(f"Total Trainable Parameters: {total_params:,}") 
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    if args.checkpoint:
        checkpoint = torch.load("/mnt/bravo/VLAT_update/VLAT/med_vlat_bert_dmcan_clef_roco_mcat_33.pth", map_location='cpu') 
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['image_encoder.vit_model.pos_embed'], model.image_encoder.vit_model)         
        state_dict['image_encoder.vit_model.pos_embed'] = pos_embed_reshaped   
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['image_encoder.vit_model.pos_embed'], model.image_encoder.vit_model)   
                state_dict['image_encoder.pos_embed'] = m_pos_embed_reshaped 
                
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')         
                    state_dict[encoder_key] = state_dict[key] 
                if 'text_encoder' in key:                
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]  
                            continue
                        else:
                            decoder_layer_num = (layer_num - 6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)     
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')  
                    state_dict[decoder_key] = state_dict[key]     
                    del state_dict[key]                
                
        msg = model.load_state_dict(state_dict, strict=False)  
        print('load checkpoint from')

    for epoch in range(start_epoch, max_epoch + 1):
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     }    

        # Log epoch statistics to wandb
        wandb.log(log_stats)
        wandb.log
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
    torch.save(save_obj, 'SLAKE_%02d.pth' % epoch)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA_Slake.yaml') 
    parser.add_argument('--checkpoint', default=True) 
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--text_encoder', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--text_decoder', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
