import argparse
import logging
import os
import pprint
import shutil
import time

import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-v", "--visualize", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument("--model_name", default="RAS", type=str, help="model name")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--seed", default=333, type=int, help="seed")
parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
parser.add_argument("--num_encoder_layers", default=2, type=int, help="num_encoder_layers")
parser.add_argument("--num_decoder_layers", default=4, type=int, help="num_decoder_layers")
parser.add_argument("--model", default="ras", type=str, help="ras")
parser.add_argument("--patchsize", default=3, type=int, help="patchify at MFCN")
parser.add_argument("--patchify_type", default="avg", type=str, help="avg, max, avg+max")
parser.add_argument("--clsname", default="all", type=str, help="clsname (unified setting: all, seperate setting: bottle, ...)")
parser.add_argument("--jitter", default=25, type=int, help="feature jitter scale")


def main():
    global args, config, key_metric, best_metric, save_ret_metrics
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)

    #update config from args
    config.dataset.batch_size = args.batch_size
    config.random_seed = args.seed
    config.net[2].kwargs.dropout = args.dropout
    config.trainer.optimizer.kwargs.lr = args.learning_rate
    config.net[2].kwargs.feature_jitter.scale = args.jitter
    config.net[2].kwargs.num_encoder_layers = args.num_encoder_layers
    config.net[2].kwargs.num_decoder_layers = args.num_decoder_layers
    config.net[1].kwargs.patchsize = args.patchsize
    config.net[1].kwargs.patchify_type = args.patchify_type
    config.dataset.clsname = args.clsname
    config.exp_path = os.path.dirname(args.config) + "/" + args.model_name
    config.save_path = os.path.join(args.model_name, config.saver.save_dir)
    config.log_path = os.path.join(args.model_name, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.save_path, config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    save_ret_metrics = None
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.save_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.save_path, load_path)
        load_state(load_path, model)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    if args.evaluate:
        validate(val_loader, model)
        return 0

    criterion = build_criterion(config.criterion)
    best_epoch = -1

    for epoch in range(last_epoch, config.trainer.max_epoch):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
        )
        lr_scheduler.step()

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model)
            # only ret_metrics on rank0 is not empty
            if rank == 0:
                ret_key_metric = (ret_metrics["mean_pixel_auc"] + ret_metrics["mean_max_auc"]) / 2
                is_best = ret_key_metric >= best_metric
                # best_metric = max(ret_key_metric, best_metric)
                if is_best:
                    best_metric = ret_key_metric
                    save_ret_metrics = ret_metrics
                    best_epoch = epoch
                
                tb_logger.add_scalar("mean_pixel_auc", ret_metrics["mean_pixel_auc"], epoch + 1)
                tb_logger.add_scalar("mean_max_auc", ret_metrics["mean_max_auc"], epoch + 1)
                tb_logger.add_scalar("pixel_at_best", save_ret_metrics["mean_pixel_auc"], epoch + 1)
                tb_logger.add_scalar("max_at_best", save_ret_metrics["mean_max_auc"], epoch + 1)
                tb_logger.flush()

                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "best_epoch": best_epoch, 
                        "arch": config.net,
                        "state_dict": model.state_dict(),
                        "best_metric": save_ret_metrics,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    config,
                )
                if best_epoch - epoch >= 100:
                    break
    
    if rank == 0 and save_ret_metrics is not None:
        log_metrics(save_ret_metrics, config.evaluator.metrics, is_best=True)


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        # current_lr = lr_scheduler.get_lr()[0]
        current_lr = optimizer.param_groups[0]['lr']

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs = model(input)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)

        print("number of data for evaluation: ", len(fileinfos), preds.shape, masks.shape)
        # evaluate, log & vis
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        if args.visualize and args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.visualize and args.evaluate and config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()
