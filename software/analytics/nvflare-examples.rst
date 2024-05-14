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

Testing 
=======
1. Run a single NVFlare app

This command will run the same hello-numpy-sag app on the server and 8 clients using 1 process. The client names will be site-1, site-2, … , site-8.

    a. Run locally with pyenv 
        i. Steps to run in virtual environment:
            1. Relocate into your projects directory
                .. code-block:: bash

                    cd [your path]

              NOTE: Do not cd into your NVFlare folder.

            2. Activate your virtual environment 
                .. code-block:: bash

                    pyenv activate nvflare

            3. Run the following command
                .. code-block:: bash

                    nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -n 8 -t 1
            
            4. Ensure the terminal output matches NVIDIA's Expected Output at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html

    b. Running on Frontier (on 2 debug nodes)
        i. Expected output should match NVIDIA's at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html
        ii. Steps to run on Frontier:

            1. SSH into Frontier using authorized email

            2. Clone the NVFlare project in a directory on Frontier
                .. code-block:: bash

                      git clone https://github.com/NVIDIA/NVFlare.git

                Note: Be sure to clone while in a Login Node.

            3. Submit your job for 30 minutes on 2 nodes: 
                .. code-block:: bash
                    
                      salloc -A <PROJECT ID> -J <JOB NAME> -t ##:##:## -p batch -N #

                    Example: salloc -A GEN007RATS -J singleNVFlareApp -t 30:00 -p batch -N 2 -q debug

                Note: For NVFLARE pilot purposes, we are testing on debug nodes.

            4. Activate your virtual environment 
                 .. code-block:: bash
                    
                      conda activate nvflare

                Note: ensure you have the correct version of NVFlare downloaded in your virtual environment

            5. Run the following simulation using NVFLARE activation command:
                .. code-block:: bash

                      nvflare simulator [-h] -w WORKSPACE [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] job_folder

              
              For example:

                .. code-block:: bash
                    
                      nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -n 8 -t 1
              
            6.  Ensure the terminal output matches NVIDIA's Expected Output at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html

2. Run a NVFlare Job
    a. Run locally with pyenv
        i. Steps to run in virtual environment 
            1. relocate into your projects directory
                .. code-block:: bash

                    cd [your path]

              NOTE: Do not cd into your NVFlare folder.
            2. Activate your virtual environment 
                .. code-block:: bash
                  
                    pyenv activate nvflare

            3. Run the following command
                .. code-block:: bash
                    
                    nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -c client0,client1,client2,client3 -t 1
            
            4. Ensure the terminal output matches NVIDIA's Expected Output at https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html


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


5. Tensorflow Open Source Example: https://github.com/bethropolis/myia

    .. note::
        This section describes the process of integrating the custom open-source MYIA model using NVIDIA examples. 
        The current example does not work with NVFLARE, but this is likely due to the chosen open-source example. 
        These steps can be used as a general guideline for integrating any model into NVFLARE. 
        In the future, consider using a backward strategy, where a working NVIDIA 
        example is used as the foundational code and all custom code is added (execute, training, etc).
  
    NOTE: NVFlare requires python version greater than 3.8

    1. Clone the MYIA repository:
        .. code-block:: bash
           
            git clone https://github.com/oliviaaheng/myia.git
    
    2. Choose your model data and take pictures for your model to train and test on
        In this walkthrough, I trained the model on writing utensils, and included non-writing utensil pictures (ie water bottle, nail polish, etc)

        NOTE: all images must be in .jpg file format
    
    3. Create a data directory: 
        Note: given that the forked repo has training, training/train, training/test, model, model/evaluation, model/labeled, model/labeled/bad, model/labeled/good 
        For the MYIA repo:
          Upload all good test images into model/evaluation/good folder 
          Upload all bad test images into model/evaluation/bad folder 
          Upload all good training images into model/labeled/good folder 
          Upload all bad training images into model/labeled/bad folder 
    
    4. Open a terminal window
        NOTE: This open-source repository requires python 3.#. (python 3.8, 3.9, 3.10 do not work)
        .. code-block::
          
            pyenv install 3.9
    
    5. Create a new virtual environment:
        .. code-block:: bash
          
            pyenv virtualenv 3.9 myiaenv
    
    6. Activate the newly created environment
        .. code-block:: bash
          
            pyenv activate myiaenv

        Note: double check your python version using:
          
        .. code-block:: bash  
          
            python --version
    
    7. Install the dependencies by running the following command:
        .. code-block:: bash
          
            pip install -r requirements.txt
        
    8. Copy and Paste the following files from NVFLARE's tf2_net example into a newly created config directory:
        .. code-block:: bash
          
            config_fed_client.json
            config_fed_server.json

    9. Copy and Paste the following files from NVFLARE's tf2_net example into a newly created custom directory:
        .. code-block:: bash
          
            pt_constants.py
            pt_model_locator.py

    10. Remove all references of simple_network.py from copied files

    11. Model your custom trainer function from NVIDIA's tf2_net trainer.py (/Users/Shared/ornldev/projects/NVFlare/examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py)
        
        NOTE: Specifically, we did the following:
          init function:
            -	Initialize custom values
            -	Setup device, custom functions (model, loss, optimizer, and scheduler, n_iterations)

          execute function
            - copied from example

          get_model_weights function
            - copied from example
          
          local_train
            - custom code for training

          save_local_model function
            - copied from example

          load_local_model function
            - copied from example


    9. Run the app in NVFlare using the following terminal command:
    
       .. code-block:: bash

                      nvflare simulator [-h] -w WORKSPACE [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] job_folder

              
       For example:
        .. code-block:: bash
            
            nvflare simulator -w /tmp/myia -n 1 -t 1 Shared/ornldev/projects/custom/app

        NOTE: 
        1. To run the command, you must be outside the path (example: Must be at Users to get to Shared)
