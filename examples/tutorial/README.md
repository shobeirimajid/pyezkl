To get started install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your system. From there create an new evironment:

```bash
conda create -n ezkl python=3.9
```

Activate your newly created environment and install the requisite dependencies:

```bash
conda activate ezkl; pip install torch numpy;       
```

The `tutorial.py` file creates a (relatively) complex Onnx graph that takes in 3 inputs `x`, `y`, and `z` and produces two outputs that we can verify against public inputs.

The file defines a computational graph as a pytorch `nn.Module` which is as follows:

```python
class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(3, 3, (2, 2), 1, 2)

        self._initialize_weights()

    def forward(self, x, y, z):
        x =  self.sigmoid(self.conv(y@x**2 + (x) - (self.relu(z)))) + 2
        return (x, self.relu(z) / 3)


    def _initialize_weights(self):
        init.orthogonal_(self.conv.weight)
```

As noted above this graph takes in 3 inputs and produces 2 outputs. The main function instantiates an instance of `Circuit` and saves it to an Onnx file.

```python
def main():
    torch_model = Circuit()
    # Input to the model
    shape = [3, 2, 2]
    x = 0.1*torch.rand(1,*shape, requires_grad=True)
    y = 0.1*torch.rand(1,*shape, requires_grad=True)
    z = 0.1*torch.rand(1,*shape, requires_grad=True)
    torch_out = torch_model(x, y, z)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x,y,z),                   # model input (or a tuple for multiple inputs)
                      "network.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    d = ((x).detach().numpy()).reshape([-1]).tolist()
    dy = ((y).detach().numpy()).reshape([-1]).tolist()
    dz = ((z).detach().numpy()).reshape([-1]).tolist()


    data = dict(input_shapes = [shape, shape, shape],
                input_data = [d, dy, dz],
                public_inputs = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump( data, open( "input.json", 'w' ) )

if __name__ == "__main__":
    main()
```
Running the file generate an `.onnx` file. Note that this also create the required input json file, whereby we use the outputs of the pytorch model as the public inputs to the circuit.

You can use these as inputs to the [ezkl](https://github.com/zkonduit/ezkl) cli as follows: 

```bash
cargo run --bin ezkl -- --scale 4 --bits 16 -K 17 table  -M ./network.onnx
```


