import io
import numpy as np
from torch import nn
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import json

class Model(nn.Module):
    def __init__(self, inplace=False):
        super(Model, self).__init__()

        self.aff1 = nn.Linear(3,4)
        self.relu1 = nn.ReLU()
        self.aff2 = nn.Linear(4,4)
        self.relu2 = nn.ReLU()
        self.aff3 = nn.Linear(4,2)
        self.relu3 = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x =  self.aff1(x)
        x =  self.relu1(x)
        x =  self.aff2(x)
        x =  self.relu2(x)
        x =  self.aff3(x)
        x =  self.relu3(x)
        return (x)


    def _initialize_weights(self):
        init.orthogonal_(self.aff1.weight)
        init.orthogonal_(self.aff2.weight)
        init.orthogonal_(self.aff3.weight)

def main():
    torch_model = Model()
    # Input to the model
    shape = [3]
    x = 0.1*torch.rand(1,*shape, requires_grad=True)
    torch_out = torch_model(x)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x,),                   # model input (or a tuple for multiple inputs)
                      "mlp.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    d = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_shapes = [shape],
                input_data = [d],
                public_inputs = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump( data, open( "input.json", 'w' ) )

if __name__ == "__main__":
    main()


    
