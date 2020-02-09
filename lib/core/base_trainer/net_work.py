#-*-coding:utf-8-*-

import torch


import time
import os

from train_config import config as cfg
#from lib.dataset.dataietr import DataIter


from lib.helper.logger import logger

from lib.core.model.ShuffleNet_Series.ShuffleNetV2.utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from lib.core.model.loss.simpleface_loss import calculate_loss


class Train(object):
  """Train class.
  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self, model,train_ds,val_ds):
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor


    if 'Adam' in cfg.TRAIN.opt:


      self.optimizer = torch.optim.Adam(get_parameters(model),
                                     lr=0.001,
                                     weight_decay=cfg.TRAIN.weight_decay_factor)

    else:
      self.optimizer  = torch.optim.SGD(get_parameters(model),
                                lr=0.001,
                                momentum=0.99,
                                weight_decay=cfg.TRAIN.weight_decay_factor)

    self.device = torch.device("cuda")

    self.model = model.to(self.device)

    ###control vars
    self.iter_num=0

    self.lr_decay_every_epoch =cfg.TRAIN.lr_decay_every_epoch
    self.lr_val_every_epoch = cfg.TRAIN.lr_value_every_epoch


    self.train_ds=train_ds

    self.val_ds = val_ds

    self.scheduler = torch.optim.lr_scheduler.MultiStepLR( self.optimizer, milestones=[60,80,100], gamma=0.1)




  def loss_function(self,predict,label):

    loss=calculate_loss(predict,label)
    return loss

  def custom_loop(self):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(epoch_num):
      total_loss = 0.0
      num_train_batches = 0.0

      self.model.train()
      for step in range(self.train_ds.size):

        start=time.time()



        images, target = self.train_ds()

        images_torch = torch.from_numpy(images)
        target_torch = torch.from_numpy(target)
        # target = target.type(torch.LongTensor)
        data, target = images_torch.to(self.device), target_torch.to(self.device)


        output = self.model(data)


        current_loss = self.loss_function(output, target)
        self.optimizer.zero_grad()
        current_loss.backward()
        self.optimizer.step()

        total_loss += current_loss
        num_train_batches += 1
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:
          logger.info('epoch_num: %d, '
                      'iter_num: %d, '
                      'loss_value: %.6f,  '
                      'speed: %d images/sec ' % (epoch_num,
                                                 self.iter_num,
                                                 current_loss,
                                                 images_per_sec))

      return total_loss, num_train_batches

    def distributed_test_epoch(epoch_num):
      total_loss=0.
      num_test_batches = 0.0
      self.model.eval()
      with torch.no_grad():
        for i in range(self.val_ds.size):
          images, target = self.val_ds()
          images_torch = torch.from_numpy(images)
          target_torch = torch.from_numpy(target)
          # target = target.type(torch.LongTensor)
          data, target = images_torch.to(self.device), target_torch.to(self.device)

          output = self.model(data)
          current_loss = self.loss_function(output, target)
          total_loss+=current_loss
          num_test_batches += 1
      return total_loss, num_test_batches


    for epoch in range(self.epochs):
      self.scheduler.step()



      start=time.time()

      train_total_loss, num_train_batches = distributed_train_epoch(epoch)
      test_total_loss, num_test_batches = distributed_test_epoch(epoch)



      time_consume_per_epoch=time.time()-start
      training_massage = 'Epoch: %d, ' \
                         'Train Loss: %.6f, ' \
                         'Test Loss: %.6f '\
                         'Time consume: %.2f'%(epoch,
                                               train_total_loss / num_train_batches,
                                               test_total_loss / num_test_batches,
                                               time_consume_per_epoch)

      logger.info(training_massage)


      #### save the model every end of epoch
      current_model_saved_name='./model/epoch_%d_val_loss%.6f.pth'%(epoch,test_total_loss / num_test_batches)

      logger.info('A model saved to %s' % current_model_saved_name)

      if not os.access(cfg.MODEL.model_path,os.F_OK):
        os.mkdir(cfg.MODEL.model_path)


      torch.save(self.model,current_model_saved_name)
      # save_checkpoint({
      #           'state_dict': self.model.state_dict(),
      #           },iters=epoch,tag=current_model_saved_name)



    return (train_total_loss / num_train_batches,
            test_total_loss / num_test_batches)





