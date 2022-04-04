# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from torch import load, save 
from Model import Model
from Trainer import Trainer
from config import model_config, dataloader_config, train_config, criterion_config, scheduler_config

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

"""
This function takes hyperparameters from config.py. 
It creates an object from the Model class and then uses it to define an object 
from the Trainer class.
The training is launched by the call to the run() method from the trainer object. 
This call is inside a try: block in order to handle exceptions.
For now, the only exception handled is a KeyboardInterrupt: 
the current network will be saved.
"""
def main(model_config, dataloader_config, train_config, criterion_config, scheduler_config):
    print('Building Model...')
    model = Model(model_config, criterion_config, scheduler_config)
    trainer = Trainer(model, dataloader_config, train_config)
    #print(trainer) #wont work with vxm
    try:
        trainer.run()
    except KeyboardInterrupt:
        #net = model.net.summary['net']
        #optimizer =  model.net.summary['optimizer']
        net='net'
        optimizer='adam'
        basename = net + '_' + optimizer + '_' + '.pt'
        filename = 'interrupted_' + basename
        path = train_config['checkpoints_path'] + filename
        save(model.net.state_dict(), path)
        print()
        print(80*'_')
        print('Training Interrupted')
        print('Current State saved.')
    path = train_config['checkpoints_path']+'training_finished.pt'
    save(model.net.state_dict(),path)
        

        
def evaluate(model_state_dict_path, model_config, dataloader_config, train_config, criterion_config, scheduler_config):
    print('Loading Model...')
    model = Model(model_config, criterion_config, scheduler_config)
    model.net.load_state_dict(load(model_state_dict_path))
    trainer = Trainer(model, dataloader_config, train_config)
    loss = trainer.test()
    print(80*'_')
    print('EVALUATING MODEL')
    print('Loss.....:  {:4f}'.format(loss))
    #print('Dice..:  {:4f}'.format(Dice))
    print(80*'_')
    return trainer
    
    

if __name__ == '__main__':
    main(model_config, dataloader_config, train_config, criterion_config, scheduler_config)
    # evaluate('./checkpoints/resnet20.pt')