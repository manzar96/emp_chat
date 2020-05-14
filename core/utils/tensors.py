from core.utils import types
from typing import cast, Callable, Optional, Tuple
import torch

def t_(data: types.NdTensor,
       dtype: torch.dtype = torch.float,
       device: Optional[types.Device] = 'cpu',
       requires_grad: bool = False) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set IN PLACE.
    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: bool): Trainable tensor or not? (Default value = False)
    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data
    """
    if isinstance(device, str):
        device = torch.device(device)

    tt = (torch.as_tensor(data, dtype=dtype, device=device)
          .requires_grad_(requires_grad))
    return tt


def t(data: types.NdTensor,
      dtype: torch.dtype = torch.float,
      device: types.Device = 'cpu',
      requires_grad: bool = False) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set. This always copies data.
    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)
    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data
    """
    tt = torch.tensor(data, dtype=dtype, device=device,
                      requires_grad=requires_grad)
    return tt


def mktensor(data: types.NdTensor,
             dtype: torch.dtype = torch.float,
             device: types.Device = 'cpu',
             requires_grad: bool = False,
             copy: bool = True) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
        is passed it is cast to  dtype, device and the requires_grad flag is
        set. This can copy data or make the operation in place.
    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)
        copy: (bool): If false creates the tensor inplace else makes a copy
            (Default value = True)
    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data
    """
    tensor_factory = t if copy else t_
    return tensor_factory(
        data, dtype=dtype, device=device, requires_grad=requires_grad)