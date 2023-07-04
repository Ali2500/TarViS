# TarViS Interactive Demo

- Complete the [environment setup](https://github.com/Ali2500/TarViS#environment-setup) in the readme. Note that you do not need to setup any directories for the demo.

- Install additional packages required for the web interface

```
pip install -r requirements_demo.txt
```

- Run the demo by providing the path to a trained checkpoint:

```
python tarvis/demo/main.py /path/to/trained/checkpoint.pth
```

Then you can view the GUI by navigating to `localhost:8050` on your web browser (port number may be differ; see on-screen output).

