from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.xbert import BertLMHeadModel, BertForMaskedLM
from models.vit import VisionTransformer, interpolate_pos_embed
from transformers import  BertConfig, BertTokenizer, CLIPVisionModel, RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForMaskedLM
torch.cuda.empty_cache()


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device)) 


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        x = self.ln(x + attn_output) # Add & Norm
        x = self.ln(x + self.ffn(x)) # FFN & Norm
        return x


class GuidedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GuidedAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_emb, text_emb):
        attn_output, _ = self.mha(image_emb, text_emb, text_emb) # Image attends to text
        image_emb = self.ln(image_emb + attn_output) # Add & Norm
        image_emb = self.ln(image_emb + self.ffn(image_emb)) # FFN & Norm
        return image_emb


class FineGrained(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=6):
        super(FineGrained, self).__init__()
        self.num_layers = num_layers
        self.sa_layers = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.ga_layers = nn.ModuleList([GuidedAttention(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, image_embeddings, text_embeddings, image_mask=None):
        # Self-attention over text embeddings (text conditioning)
        text_embeddings_sa = text_embeddings
        for layer in self.sa_layers:
            text_embeddings_sa = layer(text_embeddings_sa)

        # Iterative SA-GA processing on image embeddings
        for i in range(self.num_layers):
            image_embeddings = self.sa_layers[i](image_embeddings, image_mask) # Self-Attention
            image_embeddings = self.ga_layers[i](image_embeddings, text_embeddings_sa) # Guided Attention
        return image_embeddings, text_embeddings_sa


class ImageEncoder(nn.Module):
    def __init__(self, image_size=224, output_dim=768, init_deit=True):
        super(ImageEncoder, self).__init__()
        self.vit_model = VisionTransformer(
            img_size=image_size, patch_size=16, embed_dim=output_dim, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.vit_model)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.vit_model.load_state_dict(state_dict,strict=False) 
            print(msg)

    def forward(self, images):
        image_embeds = self.vit_model(images) 
        return image_embeds


class VLAT(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, fg_layers=6, config_file="",
                decoder_base="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        
        config_decoder = BertConfig.from_json_file(config_file)
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        
        self.image_encoder = ImageEncoder()
        bert_config = BertConfig.from_json_file(config_file)
        
        self.text_encoder = BertForMaskedLM.from_pretrained(decoder_base, config=bert_config) 
        self.fine_grained = FineGrained(embed_dim, num_heads, fg_layers)
        
        self.tokenizer = BertTokenizer.from_pretrained(decoder_base) 
        self.decoder = BertLMHeadModel.from_pretrained(decoder_base, config=config_decoder) 

    def forward(self, images, questions, answers, weights,  k, train=True, alpha=None):
        
        image_embeds = self.image_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        
        text_output = self.text_encoder.bert(questions.input_ids, attention_mask=questions.attention_mask, return_dict = True, mode = 'text') 
        text_embeds = text_output.last_hidden_state

        if train:
            answer_targets = answers.input_ids.masked_fill(answers.input_ids == self.tokenizer.pad_token_id, -100)

            fused_image_embeds, fused_text_embeds = self.fine_grained(image_embeds, text_embeds)

            fusion_outputs = self.text_encoder.bert(attention_mask = questions.attention_mask, 
                    inputs_embeds = fused_text_embeds, 
                    encoder_hidden_states = fused_image_embeds,
                    encoder_attention_mask = image_atts, 
                    return_dict = True,
                    mode = 'fusion',)
            
            fused_representation = fusion_outputs.last_hidden_state

            question_states = [] 
            question_atts = []
            for b, n in enumerate(k):
                question_states += [fused_representation[b]]*n
                question_atts += [questions.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0) 
            question_atts = torch.stack(question_atts,0)

            answer_output = self.decoder(answers.input_ids, 
                                        attention_mask = answers.attention_mask, 
                                        encoder_hidden_states = question_states,
                                        encoder_attention_mask = question_atts, 
                                        labels = answer_targets,
                                        return_dict = True,
                                        alpha=0.4, 
                                        reduction="none")

            loss = weights * answer_output.loss 
            loss = loss.sum()/images.size(0)

            return loss

        else:

            fused_image_embeds, fused_text_embeds = self.fine_grained(image_embeds, text_embeds)

            fusion_outputs = self.text_encoder.bert(attention_mask = questions.attention_mask, 
                                                    inputs_embeds = fused_text_embeds, 
                                                    encoder_hidden_states = fused_image_embeds,
                                                    encoder_attention_mask = image_atts, 
                                                    return_dict = True,
                                                    mode = 'fusion',)
            topk_ids, topk_probs = self.rank_answer(fusion_outputs.last_hidden_state, questions.attention_mask, answers.input_ids, answers.attention_mask, k) 

            return topk_ids, topk_probs



    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.decoder(start_ids, 
                                        encoder_hidden_states = question_states,
                                        encoder_attention_mask = question_atts,                                      
                                        return_dict = True, reduction = "none")              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.decoder(input_ids, 
                                attention_mask = input_atts, 
                                encoder_hidden_states = question_states,
                                encoder_attention_mask = question_atts,     
                                labels = targets_ids,
                                return_dict = True, reduction="none")                 
        answer_loss = output.logits
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    





class VLATClipRoberta(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, fg_layers=6, config_file="", 
                image_base="openai/clip-vit-base-patch16",
                text_base="roberta-base",
                decoder_base="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        # CLIP Vision Encoder
        roberta_config = RobertaConfig.from_pretrained(text_base)
        self.vision_encoder = CLIPVisionModel.from_pretrained(image_base)
        self.text_encoder = RobertaModel.from_pretrained(text_base, config=roberta_config)
        
        bert_config = BertConfig.from_json_file(config_file)
        self.fusion_block = BertForMaskedLM.from_pretrained(decoder_base, config=bert_config) 

        self.fine_grained = FineGrained(embed_dim, num_heads,fg_layers)
        
        
        config_decoder = BertConfig.from_json_file(config_file)
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6

        self.tokenizer = RobertaTokenizer.from_pretrained(text_base)
        self.decoder = BertLMHeadModel.from_pretrained(decoder_base, config=config_decoder) 

    def forward(self, images, questions, answers, weights, k, train=True, alpha=None):
        
        image_outputs = self.vision_encoder(images)
        image_embeds = image_outputs.last_hidden_state
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)
        
        # Process text through RoBERTa
        text_output = self.text_encoder(
            questions.input_ids, 
            attention_mask=questions.attention_mask,
            return_dict=True
        )
        text_embeds = text_output.last_hidden_state
        
        if train:
            # Modify padding token for RoBERTa
            answer_targets = answers.input_ids.masked_fill(
                answers.input_ids == self.tokenizer.pad_token_id, -100
            )

            fused_image_embeds, fused_text_embeds = self.fine_grained(image_embeds, text_embeds)
            # Process through RoBERTa text encoder with fusion
            fusion_outputs = self.fusion_block.bert(attention_mask = questions.attention_mask, 
                    inputs_embeds = fused_text_embeds, 
                    encoder_hidden_states = fused_image_embeds,
                    encoder_attention_mask = image_atts, 
                    return_dict = True,
                    mode = 'fusion',)
            
            
            fused_representation = fusion_outputs.last_hidden_state

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [fused_representation[b]] * n
                question_atts += [questions.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.decoder(
                answers.input_ids,
                attention_mask=answers.attention_mask,
                encoder_hidden_states=question_states,
                encoder_attention_mask=question_atts,
                labels=answer_targets,
                return_dict=True,
                alpha=0.4,
                reduction="none"
            )

            loss = weights * answer_output.loss
            loss = loss.sum() / images.size(0)
            return loss
        else:

            fused_image_embeds, fused_text_embeds = self.fine_grained(image_embeds, text_embeds)

            fusion_outputs = self.fusion_block.bert(attention_mask = questions.attention_mask, 
                                                    inputs_embeds = fused_text_embeds, 
                                                    encoder_hidden_states = fused_image_embeds,
                                                    encoder_attention_mask = image_atts, 
                                                    return_dict = True,
                                                    mode = 'fusion',)
            topk_ids, topk_probs = self.rank_answer(fusion_outputs.last_hidden_state, questions.attention_mask, 
            answers.input_ids, answers.attention_mask, k) 
            return topk_ids, topk_probs



    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.decoder(start_ids, 
                                        encoder_hidden_states = question_states,
                                        encoder_attention_mask = question_atts,                                      
                                        return_dict = True, reduction = "none")              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.decoder(input_ids, 
                                attention_mask = input_atts, 
                                encoder_hidden_states = question_states,
                                encoder_attention_mask = question_atts,     
                                labels = targets_ids,
                                return_dict = True, reduction="none")                 
        answer_loss = output.logits
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    