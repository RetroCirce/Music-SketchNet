import torch
from torch.autograd import Variable


def to_cuda_variable(tensor):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def to_cuda_variable_long(tensor):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor.long()).cuda()
    else:
        return Variable(tensor.long())


def to_numpy(variable: Variable):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :return: numpy nd array, of same size as variable
    """
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def init_hidden_lstm(num_layers, batch_size, lstm_hidden_size):
    hidden = (
        to_cuda_variable(
            torch.zeros(num_layers, batch_size, lstm_hidden_size)
        ),
        to_cuda_variable(
            torch.zeros(num_layers, batch_size, lstm_hidden_size)
        )
    )
    return hidden