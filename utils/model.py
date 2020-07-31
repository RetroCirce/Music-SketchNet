import os
import torch


class Model(torch.nn.Module):
    """
    Abstract model class
    """
    def __init__(self):
        super(Model, self).__init__()
        self.filepath = None

    def forward(self):
        pass

    def save(self):
        """
        Saves the model
        :return: None
        """
        save_dir = os.path.dirname(self.filepath)
        # create save directory if needed
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.state_dict(), self.filepath)
        print(f'Model {self.__repr__()} saved')

    def save_checkpoint(self, epoch_num):
        """
        Saves the model checkpoints
        :param epoch_num: int,
        :return: None
        """
        torch.save(self.state_dict(), self.filepath + '_' + str(epoch_num))
        print(f'Model checkpoint {self.__repr__()} saved for epoch')

    def load(self, cpu=False):
        """
        Loads the model
        :param cpu: bool, specifies if the model should be loaded on the CPU
        :return: None
        """
        if cpu:
            self.load_state_dict(
                torch.load(
                    self.filepath,
                    map_location=lambda storage,
                    loc: storage
                )
            )
        else:
            self.load_state_dict(torch.load(self.filepath))
        print(f'Model {self.__repr__()} loaded')
