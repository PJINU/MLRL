import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qd_detr.config import BaseOptions
from start_end_dataset_copy import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from qd_detr.start_end_dataset_audio import \
    StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio
from qd_detr.inference2 import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters

import ipdb

from rich.console import Console

import logging

console = Console()


# def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    
    
#     logger.info(f"[Epoch {epoch_i+1}]")
#     model.train()
#     criterion.train()

#     # init meters
#     time_meters = defaultdict(AverageMeter)
#     loss_meters = defaultdict(AverageMeter)

#     num_training_examples = len(train_loader) # 226
#     timer_dataloading = time.time()
    
    
#     for batch_idx, batch in tqdm(enumerate(train_loader),
#                                  desc="Training Iteration",
#                                  total=num_training_examples):
#         time_meters["dataloading_time"].update(time.time() - timer_dataloading)

#         timer_start = time.time()
#         if opt.a_feat_dir is None:
#             model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory) 
#             # model_inputs : ['src_txt', 'src_txt_mask', 'src_vid', 'src_vid_mask']
#             # targets : ['span_labels', 'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels']
#             # model_inputs['src_txt'] = [32, 22, 512]
#             # model_inputs['src_txt_mask'] = [32, 22]
#             # model_inputs['src_vid'] = [32, 75, 2818]
#             # model_inptus['src_vid_mask'] = [32, 75]
#         else:
#             model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
#         time_meters["prepare_inputs_time"].update(time.time() - timer_start)
#         timer_start = time.time()
#         ipdb.set_trace()        
#         outputs = model(**model_inputs) # outputs : ['pred_logits', 'pred_spans', 'saliency_scores', 'saliency_scores_neg', 'video_mask', 'aux_outputs]
#         # outputs['pred_logits'] = [32, 10, 2]
#         # outputs['pred_spans'] = [32, 10, 2]
#         # outputs['saliency_scores'] = [32, 75]
#         # outputs['saliency_scores_neg'] = [32, 75]
#         # outputs['video_mask'] = [32, 75]
        
#         loss_dict = criterion(outputs, targets)
#         # loss_dict = ['loss_span', 'loss_giou', 'loss_label', 'class_error', 'loss_saliency', 'loss_span_0', 'loss_giou_0', 'loss_label_0', 'class_error_0']
#         weight_dict = criterion.weight_dict
#         # weight_dict = {'loss_span': 10, 'loss_giou': 1, 'loss_label': 4, 'loss_saliency': 1.0, 'loss_span_0': 10, 'loss_giou_0': 1, 'loss_label_0': 4}
#         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#         time_meters["model_forward_time"].update(time.time() - timer_start)

#         timer_start = time.time()
#         optimizer.zero_grad()
#         losses.backward()
#         if opt.grad_clip > 0:
#             nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
#         optimizer.step()
#         time_meters["model_backward_time"].update(time.time() - timer_start)

#         loss_dict["loss_overall"] = float(losses)  # for logging only
#         for k, v in loss_dict.items():
#             loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

#         timer_dataloading = time.time()
#         if opt.debug and batch_idx == 3:
#             break

#     # print/add logs
#     tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
#     for k, v in loss_meters.items():
#         tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

#     to_write = opt.train_log_txt_formatter.format(
#         time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
#         epoch=epoch_i+1,
#         loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
#     with open(opt.train_log_filepath, "a") as f:
#         f.write(to_write)

#     logger.info("Epoch time stats:")
#     for name, meter in time_meters.items():
#         d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
#         logger.info(f"{name} ==> {d}")

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train(model, criterion, rep_train_loss, optimizer, optimizer_rep, lr_scheduler, lr_scheduler_rep, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    
    # ipdb.set_trace()

    if opt.a_feat_dir is None: # opt.a_feat_dir = None / a_feat_dir = audio feature
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate_audio,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )
        
    rep_train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate_audio,
        batch_size=opt.bsz_rep,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )
    
    rep_val_loader = DataLoader(
        val_dataset,
        collate_fn=start_end_collate_audio,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0 # start_epoch = 0
    else:
        start_epoch = opt.start_epoch
        
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    
    # for ijepa
    momentum = 0.996
    m_start_end = (0.996, 1)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 7
    
    # --------------- Start Moment Retrieval Train ---------------  


    for epoch_i in trange(start_epoch, opt.n_epoch, desc="MR+HL Epoch"):
        
        # Train
        if epoch_i > -1:
            # train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            logger.info(f"[MR+HL Epoch {epoch_i+1}]")
            model.train()
            # criterion.train()

            # init meters
            time_meters = defaultdict(AverageMeter)
            loss_meters = defaultdict(AverageMeter)

            num_training_examples = len(train_loader) # 226
            timer_dataloading = time.time()
            # Iteration
            
            for batch_idx, batch in tqdm(enumerate(train_loader),
                                            desc="MR+HL Training Iteration",
                                            total=num_training_examples):
                time_meters["dataloading_time"].update(time.time() - timer_dataloading)

                timer_start = time.time()
                if opt.a_feat_dir is None:
                    model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory) 
                    # model_inputs : ['src_txt', 'src_txt_mask', 'src_vid', 'src_vid_mask']
                    # targets : ['span_labels', 'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels']
                    # model_inputs['src_txt'] = [32, 22, 512]
                    # model_inputs['src_txt_mask'] = [32, 22]
                    # model_inputs['src_vid'] = [32, 75, 2818]
                    # model_inptus['src_vid_mask'] = [32, 75]
                else:
                    model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
                time_meters["prepare_inputs_time"].update(time.time() - timer_start)
                timer_start = time.time()
                # ipdb.set_trace()        
                model_inputs['train_type'] = 'mr'
                model_inputs['stage']='train'
                outputs, prediction_block, target_block = model(**model_inputs) # outputs : ['pred_logits', 'pred_spans', 'saliency_scores', 'saliency_scores_neg', 'video_mask', 'aux_outputs]
                # outputs['pred_logits'] = [32, 10, 2]
                # outputs['pred_spans'] = [32, 10, 2]
                # outputs['saliency_scores'] = [32, 75]
                # outputs['saliency_scores_neg'] = [32, 75]
                # outputs['video_mask'] = [32, 75]
                # ipdb.set_trace()
                
                loss_dict = criterion(outputs, targets)
                rep_loss = rep_train_loss(prediction_block, target_block)
                # loss_dict = ['loss_span', 'loss_giou', 'loss_label', 'class_error', 'loss_saliency', 'loss_span_0', 'loss_giou_0', 'loss_label_0', 'class_error_0']
                weight_dict = criterion.weight_dict
                # weight_dict = {'loss_span': 10, 'loss_giou': 1, 'loss_label': 4, 'loss_saliency': 1.0, 'loss_span_0': 10, 'loss_giou_0': 1, 'loss_label_0': 4}
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) + rep_loss
                time_meters["model_forward_time"].update(time.time() - timer_start)

                timer_start = time.time()
                optimizer.zero_grad()
                losses.backward()
                if opt.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer.step()
                time_meters["model_backward_time"].update(time.time() - timer_start)

                loss_dict["loss_overall"] = float(losses)  # for logging only
                for k, v in loss_dict.items():
                    loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

                timer_dataloading = time.time()
                if opt.debug and batch_idx == 3:
                    break
            
                
            # print/add logs
            tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
            for k, v in loss_meters.items():
                tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

            to_write = opt.train_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i+1,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
            with open(opt.train_log_filepath, "a") as f:
                f.write(to_write)

            logger.info("Epoch time stats:")
            for name, meter in time_meters.items():
                d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
                logger.info(f"{name} ==> {d}")
                
            lr_scheduler.step()


        
        # Evaluation
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            stop_score = metrics["brief"]["MR-full-mAP"]
                
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()



def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    pass



def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    # ipdb.set_trace()
    
    # 특정 gpu 할당
    device = torch.device(f'cuda:{opt.device_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    print('##################')
    print(opt.a_feat_dir is None)
    print(opt.a_feat_dir)
    print('##################')
    
    ipdb.set_trace()
    
    if opt.a_feat_dir is None:
        dataset_config = dict(
            dset_name=opt.dset_name, # dset_name = hl
            data_path=opt.train_path, # data/highlight_train_release.jsonl
            v_feat_dirs=opt.v_feat_dirs, # featrues/clip_features
            q_feat_dir=opt.t_feat_dir, # features/clip_text_featrues
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l, # 32
            max_v_l=opt.max_v_l, # 75
            ctx_mode=opt.ctx_mode, # video_tef
            data_ratio=opt.data_ratio, # 1.0
            normalize_v=not opt.no_norm_vfeat, # True
            normalize_t=not opt.no_norm_tfeat, # True
            clip_len=opt.clip_length, # 
            max_windows=opt.max_windows, # 5
            span_loss_type=opt.span_loss_type, # l1
            txt_drop_ratio=opt.txt_drop_ratio, # 0
            dset_domain=opt.dset_domain, # None
        )
        dataset_config["data_path"] = opt.train_path # data/higtlight_train_release.jsonl
        train_dataset = StartEndDataset(**dataset_config)
    else:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            a_feat_dir=opt.a_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset_audio(**dataset_config)


    ipdb.set_trace()
    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining    /   /features/clip_text_features
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        if opt.a_feat_dir is None:
            eval_dataset = StartEndDataset(**dataset_config)
        else:
            eval_dataset = StartEndDataset_audio(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    
    # For representation
    
    rep_train_loss = nn.L1Loss()
    optimizer_rep = torch.optim.AdamW(model.parameters(), lr=opt.lr_rep, weight_decay=opt.wd_rep)
    lr_scheduler_rep = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_rep,
        max_lr = opt.lr_rep,
        total_steps = opt.n_epoch_rep
    )
    
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, rep_train_loss, optimizer, optimizer_rep, lr_scheduler, lr_scheduler_rep, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
