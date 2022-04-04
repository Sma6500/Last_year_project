from tqdm import tqdm

import torch

from Dataloaders.registration_loader import get_dataloaders


class Trainer():
     
    def __init__(self, model, dataloader_config, train_config):
        self.model = model #from class Model, keys are : net, optimizer, device, criterion, scheduler (not mandatory)
        self.trainloader, self.testloader = self._init_dataloaders(dataloader_config)
        self.config = train_config #dict, keys are : nb_epochs, checkpoints_path, verbose
        self.state = {'train_loss': 0, 
                      'train_mean_dice': 0, 
                      'test_loss': 0, 
                      'test_mean_dice': 0, 
                      'best_mean_dice': 0,
                      'epoch': 0}
        
       
    #non fonctionnelle pour le moment
    def __str__(self): 
        title = 'Training settings  :' + '\n' + '\n'
        #net         = 'Net.......................:  ' + self.model.net + '\n' #don't work with VXM
        optimizer   = 'Optimizer.................:  ' + self.model.optimizer + '\n'
        scheduler   = 'Learning Rate Scheduler...:  ' + self.model.scheduler + '\n' #if None we print None
        nb_epochs   = 'Number of epochs..........:  ' + str(self.config['nb_epochs']) + '\n'
        summary = net + optimizer + scheduler + nb_epochs
        return (80*'_' + '\n' + summary + 80*'_')
        
    
    def _init_dataloaders(self, dataloader_config):
        return get_dataloaders(dataloader_config)
    

    def train(self):
        self.model.net.train()
        train_loss = 0
        for fixed_image, moving_image in tqdm(self.trainloader):
            
            fixed_image, moving_image = fixed_image.to(self.model.device), moving_image.to(self.model.device)
            self.model.optimizer.zero_grad()
            pred_image, flow_field = self.model.net(moving_image.float(),fixed_image.float())
            loss = self.model.criterion(pred_image, fixed_image, flow_field)
            loss.backward()
            self.model.optimizer.step()
            train_loss += loss.item()

        return train_loss
    
    
    def test(self):
        self.model.net.eval()
        test_loss = 0
        with torch.no_grad():
            for fixed_image, moving_image in tqdm(self.testloader):
                fixed_image, moving_image = fixed_image.to(self.model.device), moving_image.to(self.model.device)
                pred_image, flow_field = self.model.net(moving_image.float(),fixed_image.float())
                loss = self.model.criterion(pred_image, fixed_image, flow_field)
                test_loss += loss.item()

        return test_loss
    

    def verbose(self):
        print()
        print('Train Loss................: {:.2f}'.format(self.state['train_loss']))
        print('Test Loss.................: {:.2f}'.format(self.state['test_loss']))
        #print('Train Mean Dice............: {:.2f}'.format(self.state['train_mean_dice']))
        #print('Test Mean Dice.............: {:.2f}'.format(self.state['test_mean_dice']))
        print()
        # lr can't be read easily because ReduceOnPlateau is used
        print('Current Learning Rate.....: {:.6f}'.format(self.model.optimizer.param_groups[0]['lr']))
        #print('Best Test Mean Dice........: {:.2f}'.format(self.state['best_mean_dice']))
        
    
    def update_state(self):
        train_loss = self.train()
        test_loss = self.test()
        self.model.scheduler.step(test_loss)
        self.state['train_loss'] = train_loss
        #self.state['train_mean_dice']  = train_mean_Dice
        self.state['test_loss'] = test_loss
        #self.state['test_mean_dice'] = test_mean_Dice
        #if test_mean_Dice > self.state['best_mean_dice']:
        #    self.state['best_mean_dice'] = test_mean_Dice

            
    def run(self):
        for epoch in range(self.config['nb_epochs']):
            print(80*'_')
            print('EPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
            self.update_state()
            if self.config['verbose']:
                self.verbose()
            if epoch%self.config['checkpoint']==0:

                filename = 'checkpoint_' + str(epoch)+ '_.pt'
                path = self.config['checkpoints_path'] + filename
                torch.save(self.model.net.state_dict(), path)
                print()
                print(80*'_')
                print('Current State saved.')

    
    

