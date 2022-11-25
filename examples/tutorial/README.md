To get started install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your system. From there create an new evironment:

```bash
conda create -n ezkl python=3.9
```

Activate your newly created environment and install the requisite dependencies:

```bash
conda activate ezkl; pip install torch numpy;       
```

Make sure you are in the `pyezkl/examples/tutorial` directory. 
Then to create the `input.json` and `network.onnx` files:

```bash
python tutorial.py
```


