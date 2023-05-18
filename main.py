#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Part of the codes are borrowed or adapted from Bruno Korbar

import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from models.models import ModelBuilder
from models.imageAudio_model import ImageAudioModel
from models.imageAudioClassify_model import ImageAudioClassifyModel

from opts import get_parameters
from validate import validate
from train import train_epoch

from data import create_training_dataset, create_validation_dataset

from utils.logging import setup_logger, setup_tbx
from utils.checkpointer import Checkpointer
from utils.scheduler import default_lr_scheduler

def main(args):

    os.makedirs(args.checkpoint_path, exist_ok=True)
    # Setup logging system
    logger = setup_logger(
        "Listen_to_look, audio_preview classification",
        args.checkpoint_path,
        True
    )
    logger.debug(args)
    # Epoch logging
    epoch_log = setup_logger(
        "Listen_to_look: results",
        args.checkpoint_path, True,
        logname="epoch.log"
    )
    epoch_log.info("epoch,loss,acc,lr")

    writer = None
    if args.visualization:
        writer = setup_tbx(
            args.checkpoint_path,
            args.is_master
        )
    if writer is not None:
        logger.info("Allowed Tensorboard writer")

    # Define the model
    builder = ModelBuilder()
    '''
    fully connected linear layer that takes an input tensor of size input_dim and passes it through this layer to produce an output tensor of size num_classes.
    '''
    net_classifier = builder.build_classifierNet(512, args.num_classes).cuda()

    '''
    [Fusion Layer]

    Neural network model that takes both image and audio features as input and returns an embedding of the concatenated features. During the forward pass of the 
    ImageAudioModel, image and audio features are concatenated along the feature dimension using the torch.cat() function.
    The concatenated features are passed through two fully connected layers imageAudio_fc1 and imageAudio_fc2 with ReLU activation in between.
    The output tensor is the final embedding of the concatenated features.
    
    If weights is provided, the saved weights are loaded using a Checkpointer instance
    
    '''
    net_imageAudio = builder.build_imageAudioNet().cuda()

    '''
    building an instance of another neural network model called ImageAudioClassifyModel, which takes as input the outputs of net_imageAudio and net_classifier. 
    The concatenated output of these models is then used to make a single prediction output. 
    '''
    net_imageAudioClassify = builder.build_imageAudioClassifierNet(net_imageAudio, net_classifier, args, weights=args.weights_audioImageModel).cuda()


    '''
    Creates an instance of an LSTM-based model for audio preview using the function build_audioPreviewLSTM 
    It takes three arguments: net_imageAudioFeature, net_classifier, and args. These are:
        net_imageAudioFeature: A neural network that extracts image and audio features from input data. It could be a pre-trained model or one trained from scratch.
        net_classifier: A classifier that takes the extracted features as input and produces a probability distribution over a set of classes. 
                        This could also be a pre-trained model or one trained from scratch.
        args: A set of arguments that specifies hyperparameters and other configurations for the model.
    '''
    model = builder.build_audioPreviewLSTM(net_imageAudio, net_classifier, args)
    model = model.cuda()


    
    # DATA LOADING
    train_ds, train_collate = create_training_dataset(args,logger=logger)
    val_ds, val_collate = create_validation_dataset(args,logger=logger)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.decode_threads,
        collate_fn=train_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=val_collate
    )

    args.iters_per_epoch = len(train_loader)
    args.warmup_iters = args.warmup_epochs * args.iters_per_epoch
    args.milestones = [args.iters_per_epoch * m for m in args.milestones]

    # define loss function (criterion) and optimizer
    criterion = {}
    criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss().cuda()

    if args.freeze_imageAudioNet:
        param_groups = [{'params': model.image_queryfeature_mlp.parameters(), 'lr': args.lr},
                        {'params': model.audio_queryfeature_mlp.parameters(), 'lr': args.lr},
                        {'params': model.prediction_fc.parameters(), 'lr': args.lr},
                        {'params': model.image_key_conv1x1.parameters(), 'lr': args.lr},
                        {'params': model.audio_key_conv1x1.parameters(), 'lr': args.lr},
                        {'params': model.rnn.parameters(), 'lr': args.lr},
                        {'params': net_classifier.parameters(), 'lr': args.lr * 0.1},
                        {'params': net_imageAudio.parameters(), 'lr': 0}] 
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=1)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones)
    # make optimizer scheduler
    if args.scheduler:
        scheduler = default_lr_scheduler(optimizer, args.milestones, args.warmup_iters)

    cudnn.benchmark = True

    # setting up the checkpointing system
    write_here = True
    checkpointer = Checkpointer(model, optimizer, save_dir=args.checkpoint_path,
                                save_to_disk=write_here, scheduler=scheduler,
                                logger=logger)

    if args.pretrained_model is not None:
        logger.debug("Loading model only at: {}".format(args.pretrained_model))
        checkpointer.load_model_only(f=args.pretrained_model)

    if checkpointer.has_checkpoint():
        # call load checkpoint
        logger.debug("Loading last checkpoint")
        checkpointer.load()

    model = torch.nn.parallel.DataParallel(model).cuda()
    logger.debug(model)

    # Log all info
    if writer:
        writer.add_text("namespace", repr(args))
        writer.add_text("model", str(model))

    #
    # TRAINING
    #
    logger.debug("Entering the training loop")
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_accuracy, train_loss = train_epoch(args, epoch, train_loader, model,
                    criterion, optimizer,
                    scheduler,
                    logger, epoch_logger=epoch_log, checkpointer=checkpointer, writer=writer)

        avgpool_final_acc, lstm_final_acc, avgpool_mean_ap, lstm_mean_ap, test_loss  = validate(args, epoch, val_loader, model, criterion,
                 epoch_logger=epoch_log, writer=writer)
        if writer is not None:
            writer.add_scalars('training_curves/accuracies', {'train': train_accuracy, 'val':lstm_final_acc}, epoch)
            writer.add_scalars('training_curves/loss', {'train': train_loss, 'val':test_loss}, epoch)
    
if __name__ == '__main__':
    args = get_parameters("Listen to Look")
    main(args)