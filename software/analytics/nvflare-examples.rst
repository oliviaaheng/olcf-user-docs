*************************************************************************************
NVFlare
*************************************************************************************

Overview
========

`NVIDIA FLARE  <https://nvflare.readthedocs.io/en/2.3/flare_overview.html>`_ (NVIDIA Federated Learning Application Runtime Environment) 
is a domain-agnostic, open-source, extensible SDK that allows researchers, data scientists and data engineers to adapt existing ML/DL 
and compute workflows to a federated paradigm.
NVFlare GitHub: https://github.com/NVIDIA/NVFlare
NOTE: This page follows NVFlare Version '2.3'


Getting Started
===============
1. Install NVFlare on your machine using https://github.com/NVIDIA/NVFlare?tab=readme-ov-file#installation
2. Set-up NVFlare for ORNL's Frontier using https://code.ornl.gov/q8s/hpc-fl/-/tree/longcoyaj-nvflare-update?ref%5C_type=heads
.. run first example local and then try on frontier 
.. get a new ml model training that isnt made by NVIDIA and customize to work w nvflare
.. 2 debug nodes 

Testing 
=======
1. Run a single NVFlare app
This command will run the same hello-numpy-sag app on the server and 8 clients using 1 process. The client names will be site-1, site-2, … , site-8:
    a. Run locally with pyenv
        i. Steps to run in virtual environment 
            1. relocate into your projects directory
                .. code-block:: bash
                  cd [your path]
              NOTE: Do not cd into your NVFlare folder.
            2. activate your virtual environment 
                .. code-block:: bash
                  pyenv activate nvflare
            3. Run the following command
                .. code-block:: bash
                  nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -n 8 -t 1
            4. Ensure the terminal output matches NVIDIA's Expected Output at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html

    b. Running on Frontier (on 2 debug nodes)
        i. expected output should match NVIDIA's at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html
        ii. Steps to run on Frontier:
            1. ssh into Frontier using your ornl email
            2. Clone the NVFlare project in a directory on Frontier
                .. code-block:: bash
                    git clone https://github.com/NVIDIA/NVFlare.git
                Note: Be sure to clone while in a Login Node.
            3. Submit your job for 30 minutes on 2 nodes: 
                .. code-block:: bash
                    salloc -A <PROJECT ID> -J <JOB NAME> -t ##:##:## -p batch -N #

                    Example: salloc -A GEN007RATS -J singleNVFlareApp -t 30:00 -p batch -N 2 -q debug
                Note: For nvflare pilot purposes, we are testing on debug nodes.
            4. activate your virtual environment 
                 .. code-block:: bash
                    conda activate nvflare
                Note: ensure you have the correct version of NVFlare downloaded in your virtual environment
            5. Run the following simulation
                .. code-block:: bash
                    nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -n 8 -t 1
                Recieved the following error:
Exception in thread Thread-36:
Traceback (most recent call last):
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/private/fed/app/simulator/simulator_runner.py", line 423, in start_server_app
    server_app_runner.start_server_app(
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/private/fed/server/server_app_runner.py", line 83, in start_server_app
    raise e
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/private/fed/server/server_app_runner.py", line 59, in start_server_app
    conf = ServerJsonConfigurator(config_file_name=server_config_file_name, args=args, kv_list=kv_list)
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/private/fed/server/server_json_config.py", line 59, in __init__
    FedJsonConfigurator.__init__(
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/private/fed_json_config.py", line 40, in __init__
    JsonConfigurator.__init__(
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/private/json_configer.py", line 75, in __init__
    self.module_scanner = ModuleScanner(base_pkgs, module_names, exclude_libs)
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/fuel/utils/class_utils.py", line 89, in __init__
    self._create_classes_table()
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/fuel/utils/class_utils.py", line 104, in _create_classes_table
    module = importlib.import_module(module_name)
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 843, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/app_opt/pt/__init__.py", line 15, in <module>
    from nvflare.app_opt.pt.ditto import PTDittoHelper
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/nvflare/app_opt/pt/ditto.py", line 18, in <module>
    import torch
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/torch/__init__.py", line 228, in <module>
    _load_global_deps()
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/torch/__init__.py", line 187, in _load_global_deps
    raise err
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/site-packages/torch/__init__.py", line 168, in _load_global_deps
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
  File "/ccs/home/olivia/miniconda_frontier/envs/nvflare/lib/python3.8/ctypes/__init__.py", line 373, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libgomp.so: cannot open shared object file: No such file or directory
2024-02-15 15:18:26,599 - SimulatorRunner - ERROR - Simulator run error: RuntimeError: Could not start the Server App.
2024-02-15 15:18:30,483 - MPM - INFO - MPM: Good Bye!
            #.  xxx
            #.  Ensure the terminal output matches NVIDIA's Expected Output at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html

2. Run a NVFlare Job
    a. Run locally with pyenv
        i. Steps to run in virtual environment 
            1. relocate into your projects directory
                .. code-block:: bash
                  cd [your path]
              NOTE: Do not cd into your NVFlare folder.
            2. activate your virtual environment 
                .. code-block:: bash
                  pyenv activate nvflare
            3. Run the following command
                .. code-block:: bash
                  nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -c client0,client1,client2,client3 -t 1
            4. Ensure the terminal output matches NVIDIA's Expected Output at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html

    b. Frontier (on 2 debug nodes)
        i. status: waiting for more space

3. Hello World Examples
NVIDIA provides several examples to help you get started using federated learning for your own applications. https://github.com/NVIDIA/NVFlare/tree/main/examples#1-hello-world-examples
    a. Download the Notebook at https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/hello_world.ipynb
    b. Create a new virtual environment for examples
           .. code-block:: bash
              python3 -m venv nvflare-example
    c. Hello Scatter and Gather (https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/hello-numpy-sag/README.md#hello-numpy-scatter-and-gather)
        i. Run the following command
            .. code-block:: bash
              pip install --upgrade pip
              pip install -r requirements.txt
        ii. USES NOTEBOOK, STILL TEST??

4. Step-by-Step Examples
NVIDIA provides several examples to help you get started using federated learning for your own applications. https://github.com/NVIDIA/NVFlare/tree/main/examples#2-step-by-step-examples
    a. image_stats
        i. cd into cifar10<stats folder
            .. code-block:: bash
                  cd [your path]/NVFlare/examples/hello-world/step-by-step/cifar10/stats
        ii. Run the following command
            .. code-block:: bash
                  pip install -r requirements.txt
        iii. Run the following command
            .. code-block:: bash
                  python ../data/download.py

    b. sag
        i. 
    c. sag_deploy_map
        i. 

5. Open Source Example: https://github.com/bethropolis/myia
    1. Clone the MYIA repository:
        .. code-block:: bash
            git clone https://github.com/bethropolis/myia.git
    2. Choose your model data and take pictures for your model to train and test on
        In this walkthrough, I trained the model on writing utensils, and included non-writing utensil pictures (ie water bottle, nail polish, etc)
    3. Upload all test and training images into the respective test and training folder in the myia folder
    4. Open a terminal window
    5. Create a new virtual environment:
        .. code-block:: bash
            python3 -m venv myiaenv
    6. Activate the newly created environment
        .. code-block:: bash
            source myiaenv/bin/activate 
    7. Install the dependencies by running the following command:
        .. code-block:: bash
             pip install -r requirements.txt
        Note: I recieved the following error:
        .. code-block:: bash
            ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11
            ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0.post1 (from versions: 2.12.0rc0, 2.12.0rc1, 2.12.0, 2.12.1, 2.13.0rc0, 2.13.0rc1, 2.13.0rc2, 2.13.0, 2.13.1, 2.14.0rc0, 2.14.0rc1, 2.14.0, 2.14.1, 2.15.0rc0, 2.15.0rc1, 2.15.0)
            ERROR: No matching distribution found for tensorflow==2.15.0.post1
        To resolve, I ran 
            .. code-block:: bash
                pip3 install --upgrade pip
                pip3 install tensorflow==2.15.0.post1 (DIDNT WORK)
                    ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0.post1 (from versions: 2.12.0rc0, 2.12.0rc1, 2.12.0, 2.12.1, 2.13.0rc0, 2.13.0rc1, 2.13.0rc2, 2.13.0, 2.13.1, 2.14.0rc0, 2.14.0rc1, 2.14.0, 2.14.1, 2.15.0rc0, 2.15.0rc1, 2.15.0)
                    ERROR: No matching distribution found for tensorflow==2.15.0.post1
                pip install -r requirements.txt
                pyenv install 3.8
                pip install -r requirements.txt


    8. Setup project:
        .. code-block:: bash
            python setup.py 
.. note::
..     The `spark documentation <https://spark.apache.org/docs/latest/>`_ is very useful tool, go through it to find the Spark capabilities.
