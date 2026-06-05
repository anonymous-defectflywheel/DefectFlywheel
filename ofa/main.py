import VVCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import torch.nn.init as init
from torch.optim import lr_scheduler
import prompt_generator

import torch.nn as nn
from prompt_ensemble import encode_text_with_prompt_ensemble
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import open_clip
from torch import optim

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from utils import aug

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    loss = 0
    for item in range(len(a)):
        loss += torch.dot(a[item], b[item]) / (torch.sqrt(torch.sum(a[item]**2)) * torch.sqrt(torch.sum(b[item]**2)))
    return loss / len(a)




from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def main(args):
    img_size = args.image_size
    features_list = args.features_list


    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #this parameter are not used in our model, they just use to make VVCLIP to be built successfully.
    VVCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    img_size = 224

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", img_size, pretrained="openai")

    #introduing VV-attention mechanism ONLY use its visual encoder
    model, _ = VVCLIP_lib.load("ViT-L-14", device=device, design_details = VVCLIP_parameters)

    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model.eval()
    model.eval()
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    #introduing Q-former from BLIP-diffusion
    blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
        "/home/lwx/model_card", torch_dtype=torch.float32
    ).to(device)

    results = {}
    metrics = {}

    
    model.to(device)
    clip_model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    
    #few-shot learning parameter
    shot = 1
    epochs = 20

    padding = tokenizer("").to(device)
    repersent_vec = {}
    visual_feature_bank_1 = {}
    visual_feature_bank_2 = {}
    soft_prompt_list = {}
    optimizer_list = {}
    cos_loss = nn.CosineSimilarity(dim=2)
    criterion = nn.CrossEntropyLoss().to(device)
    best_pixel_auroc = 0
    best_result = None

    #obtain embedding of manual prompt
    with torch.no_grad():
        text_prompts, text_prompts_list = encode_text_with_prompt_ensemble(clip_model, ['object'], tokenizer, device, dataset = args.dataset)

    soft_prompt = prompt_generator.SoftPrompt()
    soft_prompt = soft_prompt.to(device)

    optimizer = optim.Adam(soft_prompt.parameters(), 1e-5, weight_decay=1e-4)
    scheduler=lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=0.05)

    #training
    for epoch in range(epochs):
        print(epoch)
        results = {}
        metrics = {}
        random.shuffle(obj_list)
        for obj in obj_list:
            print(obj)
            results[obj] = {}
            results[obj]['gt_sp'] = []
            results[obj]['pr_sp'] = []
            results[obj]['imgs_masks'] = []
            results[obj]['anomaly_maps'] = []
            metrics[obj] = {}
            metrics[obj]['pixel-auroc'] = 0
            metrics[obj]['pixel-aupro'] = 0
            metrics[obj]['image-auroc'] = 0
            metrics[obj]['image-ap'] = 0

            if args.dataset == 'mvtec':
                data_path = '/home/lwx/mvtec/' + obj + '/train'
            else:
                data_path = '/home/lwx/visa4ad/1cls/' + obj + '/train/good/'
            
            #fetch few-shot data
            for i in range(shot):
                if args.dataset == 'mvtec':
                    cond_image = load_image(data_path + '/good/00' + str(i * 3) + '.png')
                else:
                    cond_image = load_image(data_path + '000' + str(i * 3) + '.JPG')                
                reference_image = blip_diffusion_pipe.image_processor.preprocess(
                            cond_image, do_resize=True, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                        )["pixel_values"]
                reference_image = reference_image.to(device)


                with torch.no_grad():
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)
                    image_embedding, patch_embedding, _ = model.encode_image(reference_image, features_list, DPAM_layer = 20, ffn=False)
                    image_embedding = image_embedding.mean(dim=0).unsqueeze(dim=0)
                    image_embedding = image_embedding / image_embedding.norm()
                    
                pos_query, neg_query = soft_prompt(query, image_embedding)
                pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_query, padding, device)
                neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_query, padding, device)

                pos_token = pos_token / pos_token.norm(dim = -1, keepdim = True)
                neg_token = neg_token / neg_token.norm(dim = -1, keepdim = True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm(dim = -1, keepdim = True)
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm(dim = -1, keepdim = True)

                # text-prompt alignment
                p_ptext_sim = torch.dot(pos_query_embedding[0], text_prompts['object'][:,0]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                n_ptext_sim = torch.dot(neg_query_embedding[0], text_prompts['object'][:,0]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                p_ntext_sim = torch.dot(pos_query_embedding[0], text_prompts['object'][:,1]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                n_ntext_sim = torch.dot(neg_query_embedding[0], text_prompts['object'][:,1]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                text_loss = ((1 - p_ptext_sim) + (1 - n_ntext_sim) + n_ptext_sim + p_ntext_sim) / 4
                
                patch_loss = 0
                fg_pos_it_patch = 0
                fg_pos_ti_patch = 0
                fg_neg_it_patch = 0
                fg_neg_ti_patch = 0
                thhold = 1 / 256


                for i in range(4):
                    noise = torch.randn_like(patch_embedding[i]) * 2
                    patch_embedding[i] = patch_embedding[i] / patch_embedding[i].norm(dim = -1, keepdim = True)

                    neg_patch_embedding = patch_embedding[i - 1] + patch_embedding[i] + noise
                    neg_patch_embedding = neg_patch_embedding / neg_patch_embedding.norm(dim = -1, keepdim = True)
                    
                    # patch-prompt alignment
                    p_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], pos_query_embedding[0]).mean().mean()
                    n_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], neg_query_embedding[0]).mean().mean()
                    n_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], neg_query_embedding[0]).mean().mean()
                    p_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], pos_query_embedding[0]).mean().mean()
                    patch_loss += ((1 - p_ppatch_sim) + (1 - n_npatch_sim) + n_ppatch_sim + p_npatch_sim) / 4

                    # patch-token alignment
                    pos_similarity = torch.einsum('btd,bpd->btp', pos_token, patch_embedding[i][:, 1:, :])
                    pos_similarity = (pos_similarity - torch.min(pos_similarity, dim = -1, keepdim = True)[0]) / (torch.max(pos_similarity, dim = -1, keepdim = True)[0] - torch.min(pos_similarity, dim = -1, keepdim = True)[0])
                    pos_similarity = torch.where(pos_similarity < thhold, 0.0, pos_similarity)
                    pos_weights = pos_similarity / torch.sum(pos_similarity, dim=-1).T
                    pos_group_embed = torch.einsum('btp,bpd->btd', pos_weights, patch_embedding[i][:, 1:, :])
                    pos_group_embed = pos_group_embed / pos_group_embed.norm(dim = -1, keepdim = True)

                    pos_it_logits = torch.einsum('btd,bpd->btp', pos_group_embed, pos_token).squeeze(dim=0)
                    pos_it_labels = torch.eye(pos_it_logits.shape[1]).to(device)
                    pos_ti_logits = torch.einsum('btd,bpd->btp', pos_token, pos_group_embed).squeeze(dim=0)
                    pos_ti_labels = torch.eye(pos_ti_logits.shape[1]).to(device)
                    fg_pos_it_patch += criterion(pos_it_logits, pos_it_labels)
                    fg_pos_ti_patch += criterion(pos_ti_logits, pos_ti_labels)

                    neg_similarity = torch.einsum('btd,bpd->btp', neg_token, neg_patch_embedding[:, 1:, :])
                    neg_similarity = (neg_similarity - torch.min(neg_similarity, dim = -1, keepdim = True)[0]) / (torch.max(neg_similarity, dim = -1, keepdim = True)[0] - torch.min(neg_similarity, dim = -1, keepdim = True)[0])
                    neg_similarity = torch.where(neg_similarity < thhold, 0.0, neg_similarity)
                    neg_weights = neg_similarity / torch.sum(neg_similarity, dim=-1).T

                    neg_group_embed = torch.einsum('btp,bpd->btd', neg_weights, neg_patch_embedding[:, 1:, :])
                    neg_group_embed = neg_group_embed / neg_group_embed.norm(dim = -1, keepdim = True)

                    neg_it_logits = torch.einsum('btd,bpd->btp', neg_group_embed, neg_token).squeeze(dim=0)
                    neg_it_labels = torch.eye(neg_it_logits.shape[1]).to(device)
                    neg_ti_logits = torch.einsum('btd,bpd->btp', neg_token, neg_group_embed).squeeze(dim=0)

                    neg_ti_labels = torch.eye(neg_ti_logits.shape[1]).to(device)
                    fg_neg_it_patch += criterion(neg_it_logits, neg_it_labels)
                    fg_neg_ti_patch += criterion(neg_ti_logits, neg_ti_labels)
                    
                patch_loss /= 4.0
                fg_pos_it_patch /= 4.0
                fg_pos_ti_patch /= 4.0
                fg_neg_it_patch /= 4.0
                fg_neg_ti_patch /= 4.0
                fg_loss = (fg_pos_it_patch + fg_pos_ti_patch + fg_neg_it_patch + fg_neg_ti_patch) / 4.0
                
                loss = text_loss + 0.125 * patch_loss + fg_loss 
                print(text_loss.cpu(), patch_loss.cpu(), fg_loss.cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scheduler.step()
        
        #building memory bank
        with torch.no_grad():
            shot_represents_bank = []
            pos_shot_represents = []
            neg_shot_represents = []
            visual_feature_bank_1['object'] = []
            visual_feature_bank_2['object'] = []
            query_bank = []
            for obj in obj_list:
                print(obj)
                if args.dataset == 'mvtec':
                    data_path = '/home/lwx/mvtec/' + obj + '/train'
                else:
                    data_path = '/home/lwx/visa4ad/1cls/' + obj + '/train/good/'
                shot_represents = []
                for i in range(shot):
                    if args.dataset == 'mvtec':
                        cond_image = load_image(data_path + '/good/00' + str(i * 3) + '.png')
                    else:
                        cond_image = load_image(data_path + '000' + str(i * 3) + '.JPG')
                    reference_image = blip_diffusion_pipe.image_processor.preprocess(
                                cond_image, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                            )["pixel_values"]
                    reference_image = reference_image.to(device)
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)
                    image_embedding, patch_embedding, patch_token_memory = model.encode_image(reference_image, features_list, DPAM_layer = 20, ffn=False)
                    image_embedding = image_embedding.mean(dim=0).unsqueeze(dim=0)
                    pos_prompt_query, neg_prompt_query = soft_prompt(query, image_embedding)
                    pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                    neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

                    visual_feature_bank_1['object'].append(patch_token_memory[0][0][1:])
                    visual_feature_bank_2['object'].append(patch_token_memory[2][0][1:])
  
                    pos_shot_represents.append(pos_query_embedding)
                    neg_shot_represents.append(neg_query_embedding)
                    query /= query.norm(dim=-1, keepdim=True)
                    query_bank.append(query)

                    shot_represents_bank.append(pos_query_embedding)
                    shot_represents_bank.append(neg_query_embedding)

            visual_feature_bank_1['object'] = torch.stack(visual_feature_bank_1['object'], dim=0)
            visual_feature_bank_2['object'] = torch.stack(visual_feature_bank_2['object'], dim=0)
            visual_feature_bank_1['object'] = F.normalize(visual_feature_bank_1['object'], dim=-1)
            visual_feature_bank_2['object'] = F.normalize(visual_feature_bank_2['object'], dim=-1)

            shot_represents_bank = torch.stack(shot_represents_bank,dim=0).view(-1, 2, 768)
            query_bank = torch.vstack(query_bank)
            shot_represents_bank /= shot_represents_bank.norm(dim = -1, keepdim = True)

            pos_shot_represents = torch.vstack(pos_shot_represents)
            neg_shot_represents = torch.vstack(neg_shot_represents)

            pos_shot_represents = pos_shot_represents.mean(dim = 0)
            neg_shot_represents = neg_shot_represents.mean(dim = 0)
            pos_shot_represents /= pos_shot_represents.norm(dim = -1, keepdim = True)
            neg_shot_represents /= neg_shot_represents.norm(dim = -1, keepdim = True)
            shot_represents = text_prompts['object'].clone().T
            shot_represents[0] = pos_shot_represents
            shot_represents[1] = neg_shot_represents
            shot_represents = shot_represents.T


            


        #testing 
        model.to(device)
        for idx, items in enumerate(test_dataloader):
            image = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
            results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
            with torch.no_grad():
                image_features, patch_features, patch_token_memory = model.encode_image(image, features_list, DPAM_layer = 20, ffn=False)                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                query = blip_diffusion_pipe.get_query_embeddings(image, ['object']*10)
                query = query.mean(dim=0).unsqueeze(dim=0)

                pos_prompt_query, neg_prompt_query = soft_prompt(query, image_features)

                pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

                pos_token = pos_token / pos_token.norm(dim = -1, keepdim = True)
                neg_token = neg_token / neg_token.norm(dim = -1, keepdim = True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm()
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm()

                query /= query.norm(dim=-1, keepdim=True)
                query = query.expand(len(obj_list) * shot, -1, -1)
                query_sim = torch.mean(torch.sum(torch.mul(query_bank, query), dim=-1), dim = -1)
                
                obj_idx = torch.topk(query_sim, k=shot)[1].cpu().numpy().tolist()
                cur_visual_feature_bank_1 = visual_feature_bank_1['object'][obj_idx].view(-1, 1024)
                cur_visual_feature_bank_2 = visual_feature_bank_2['object'][obj_idx].view(-1, 1024)


                text_features = shot_represents_bank.clone()

                text_probs = torch.matmul(text_features, image_features.T).permute(0,2,1)
                text_probs = (text_probs).softmax(-1).view(len(obj_list) * shot, -1)
                text_probs = text_probs[obj_idx] + query_sim[obj_idx].view(len(obj_idx), -1)

                text_probs = text_probs.softmax(0)

                cur_text_features = text_features[obj_idx].clone().mean(dim=0)

                cur_text_features[0] = text_features[obj_idx][:, 0, :].clone().mean(dim=0)
                cur_text_features[1] = text_features[obj_idx][:, 1, :].clone().mean(dim=0)





                text_probs = torch.matmul(text_features, image_features.T).permute(0,2,1)
                text_probs = (text_probs/0.07).softmax(-1).view(len(obj_list) * shot, -1)
                text_probs = text_probs[obj_idx].mean(dim=0)

                cur_text_features[0] = (cur_text_features[0].unsqueeze(dim=0) + pos_query_embedding) / 2
                cur_text_features[1] = (cur_text_features[1].unsqueeze(dim=0) + neg_query_embedding) / 2


                cur_text_features[0] = cur_text_features[0] / cur_text_features[0].norm()
                cur_text_features[1] = cur_text_features[1] / cur_text_features[1].norm()
                cur_text_features = cur_text_features.T
                
                anomaly_map_list = []

                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                        similarity, _ = VVCLIP_lib.compute_similarity(patch_feature, cur_text_features.T)
                        similarity_map = similarity[:, 1:, :]
                        similarity_map = similarity_map.reshape(similarity_map.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
                        similarity_map = similarity_map.permute(0, 2, 3, 1)
                        anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                        anomaly_map_list.append(anomaly_map)

                        
                vis_feature_1 = patch_token_memory[0][0][1:]
                vis_feature_1 = vis_feature_1 / vis_feature_1.norm(dim=-1, keepdim=True)
                vis_feature_2 = patch_token_memory[2][0][1:]
                vis_feature_2 = vis_feature_2 / vis_feature_2.norm(dim=-1, keepdim=True)

                score1, _ = (1.0 - vis_feature_1 @ cur_visual_feature_bank_1.t()).min(dim=-1)
                score1 /= 2.0

                score2, _ = (1.0 - vis_feature_2 @ cur_visual_feature_bank_2.t()).min(dim=-1)
                score2 /= 2.0
                score = score1 + score2
                vis_score = score.reshape(1,1,16,16)
                
                anomaly_map = torch.stack(anomaly_map_list)
                textual_anomaly_map = anomaly_map.sum(dim = 0) / 4.0
                textual_anomaly_map = textual_anomaly_map.reshape(1,1,16,16)

                anomaly_map = textual_anomaly_map + vis_score 

                text_probs = -text_probs[0].unsqueeze(dim=0) - query_sim[obj_idx].mean(dim=0) + torch.max(textual_anomaly_map) + torch.max(vis_score)

                text_probs = text_probs.view(1)

                anomaly_map = F.interpolate(anomaly_map, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
                results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

        table_ls = []
        image_auroc_list = []
        image_ap_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []
        for obj in obj_list:
            table = []
            table.append(obj)
            results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
            if args.metrics == 'image-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
            elif args.metrics == 'pixel-level':
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            elif args.metrics == 'image-pixel-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            table_ls.append(table)

        if args.metrics == 'image-level':
            # logger
            table_ls.append(['mean', 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
        elif args.metrics == 'pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                        ])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
        elif args.metrics == 'image-pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
        if np.mean(pixel_auroc_list) * 100 > best_pixel_auroc:
            best_pixel_auroc = np.mean(pixel_auroc_list) * 100
            best_result = results
        print(best_pixel_auroc, results)
    logger.info("\n%s", best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VVCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)
