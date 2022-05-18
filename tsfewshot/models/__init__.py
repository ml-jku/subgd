import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BaseModel, BasePytorchModel, MetaCurvatureWrapper, MetaSGDWrapper
from tsfewshot.models.cnn import CNN
from tsfewshot.models.feedforward import FeedForward
from tsfewshot.models.imagecnn import ImageCNN
from tsfewshot.models.lstm import LSTM
from tsfewshot.models.manuallstm import ManualLSTM
from tsfewshot.models.rnn import RNN
from tsfewshot.models.resnet import Resnet
from tsfewshot.models.eulerode import EulerODE
from tsfewshot.models.sklearnmodels import Linear, Persistence, SupportMean


def get_model(cfg: Config, is_test: bool = False, is_finetune: bool = False) -> BaseModel:
    """Get model by name.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_test : bool, optional
        Indicates whether the model will be used for training or testing.
    is_finetune : bool, optional
        Indicates whether the model will be used for normal training or for finetuning on a target task.

    Returns
    -------
    BaseModel
        Model as specified in the run configuration.
    """
    if is_finetune and not is_test:
        raise ValueError('Conflicting arguments is_test and is_finetune.')

    if cfg.model == 'lstm':
        if is_finetune:
            if cfg.lstm_finetune_error_input or cfg.lstm_head_inputs['finetune'] != ['h'] \
                    or any(v != 1 for v in cfg.lstm_n_gates['finetune'].values()):
                # model needs access to gates/states or has multiple forget/output gates
                # or we need the implementation that has additional weights to ingest the error on previous timesteps.
                # In all cases, we need a custom-built LSTM.
                model = ManualLSTM(cfg, is_finetune=is_finetune)
            else:
                model = LSTM(cfg)
        else:
            if cfg.lstm_head_inputs['train'] != ['h'] or any(v != 1 for v in cfg.lstm_n_gates['train'].values()):
                # model needs access to gates/states or has multiple forget/output gates.
                # In both cases, we need a custom-built LSTM.
                model = ManualLSTM(cfg, is_finetune=is_finetune)
            else:
                model = LSTM(cfg)
    elif cfg.model == 'rnn':
        model = RNN(cfg)
    elif cfg.model == 'cnn':
        model = CNN(cfg)
    elif cfg.model == 'imagecnn':
        model = ImageCNN(cfg)
    elif cfg.model == 'feedforward':
        model = FeedForward(cfg)
    elif cfg.model == 'persistence':
        model = Persistence(cfg)
    elif cfg.model == 'support-mean':
        model = SupportMean(cfg)
    elif cfg.model == 'linear':
        model = Linear(cfg)
    elif cfg.model == 'resnet':
        model = Resnet(cfg)
    elif cfg.model == 'eulerode':
        model = EulerODE(cfg)
    else:
        raise ValueError(f'Unknown model {cfg.model}')

    if isinstance(model, BasePytorchModel) and not isinstance(model, EulerODE):
        if is_test and (not cfg.layer_per_dataset_eval
                        or (cfg.layer_per_dataset_eval and cfg.layer_per_dataset == 'output')):
            # make sure we only use one input/output network during inference
            model.input_layer.use_n_random_nets = 1
            model.head.use_n_random_nets = 1
        else:
            if cfg.layer_per_dataset in ['mixedrotation', 'singlerotation']:
                model.lstm.input_size = model.input_layer.output_size * len(cfg.train_datasets)  # type: ignore
                model.lstm.weight_ih_l0 = nn.Parameter(torch.zeros((model.lstm.weight_ih_l0.shape[0],  # type: ignore
                                                                    model.input_layer.output_size
                                                                    * len(cfg.train_datasets))).to(model.device))
                nn.init.uniform_(model.lstm.weight_ih_l0,  # type: ignore
                                 -torch.sqrt(torch.tensor(1 / cfg.hidden_size)),  # type: ignore
                                 torch.sqrt(torch.tensor(1 / cfg.hidden_size)))  # type: ignore
            elif cfg.layer_per_dataset in ['output']:
                head = model.head.networks[0].fc[0]
                head.out_features = len(cfg.train_datasets) * head.out_features
                head.weight = nn.Parameter(torch.zeros((head.weight.shape[0]  # type: ignore
                                                        * len(cfg.train_datasets),
                                                        head.weight.shape[1])).to(model.device))
                head.bias = nn.Parameter(torch.zeros((head.bias.shape[0]  # type: ignore
                                                      * len(cfg.train_datasets))).to(model.device))
                nn.init.uniform_(head.weight,  # type: ignore
                                 -torch.sqrt(torch.tensor(1 / head.weight.shape[1])),  # type: ignore
                                 torch.sqrt(torch.tensor(1 / head.weight.shape[1])))  # type: ignore
                nn.init.uniform_(head.bias,  # type: ignore
                                 -torch.sqrt(torch.tensor(1 / head.bias.shape[0])),  # type: ignore
                                 torch.sqrt(torch.tensor(1 / head.bias.shape[0])))  # type: ignore
    else:
        if cfg.layer_per_dataset is not None:
            raise ValueError('Cannot use per-dataset input/output layers on non-PyTorch models.')

    if cfg.training_setup == 'metasgd':
        if not isinstance(model, BasePytorchModel):
            raise ValueError('MetaSGD model must be a PyTorch model.')
        model = MetaSGDWrapper(cfg, model)

    if cfg.training_setup == 'metacurvature':
        if not isinstance(model, BasePytorchModel):
            raise ValueError('Meta-Curvature model must be a PyTorch model.')
        model = MetaCurvatureWrapper(cfg, model)

    return model
