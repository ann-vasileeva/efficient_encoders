
This repository is heavily based on StyleFeatureEditor: https://github.com/ControlGenAI/StyleFeatureEditor


## Repository structure

      .
      ├── 📂 arguments                  # Contains all arguments used in training and inference
      ├── 📂 assets                     # Folder with method preview and example images
      ├── 📂 configs                    # Includes configs (associated with arguments) for training and inference
      ├── 📂 criteria                   # Contains original code for used losses and metrics
      ├── 📂 datasets                   
      │   ├── 📄 datasets.py                # A branch of custom datasets 
      │   ├── 📄 loaders.py                 # Custom infinite loader
      │   └── 📄 transforms.py              # Transforms used in SFE
      │
      ├── 📂 editings                   # Includes original code for various editing methods and an editor that applies them
      │   ├── ...
      │   └── 📄 latent_editor.py           # Implementation of module that edits w or stylespace latents 
      │
      ├── 📂 metrics                    # Contains wrappers over original code for all used inversion metrics
      ├── 📂 models                     # Includes original code from several previous inversion methods 
      │   ├── ...
      │   ├── 📂 farl                       # Modified FARL module, used to search face mask
      │   ├── 📂 psp
      │   │   ├── 📂 encoders                   # Contains all the Inverter, Feature Editor and E4E parts
      │   │   └── 📂 stylegan2                  # Includes modified StyleGAN 2 generator 
      │   │ 
      │   └── 📄 methods.py                  # Contains code for Inverter and Feature Editor modules
      │   
      ├── 📂 notebook                   # Folder for Jupyter Notebook and raw images
      ├── 📂 runners                    # Includes main code for training and inference pipelines
      ├── 📂 scripts                    # Script to ...
      │   ├── 📄 align_all_parallel.py       # Align raw images 
      │   ├── 📄 calculate_metrics.py        # Inversion metrics calculation
      │   ├── 📄 fid_calculation.py          # Editing metric calculation
      │   ├── 📄 dichotomy_sfe.py             # Count best possible FID for direction and method
      │   ├── 📄 inference.py                # Inference large set of data with several directions
      │   ├── 📄 simple_inference.py         # Inference single image with one direction and mask
      │   └── 📄 train.py                    # Start training process 
      │   
      ├── 📂 training                   
      │   ├── 📄 loggers.py                  # Code for loggers used in training
      │   ├── 📄 losses.py                   # Wrappers over used losses  
      │   └── 📄 optimizers.py               # Wrappers over used optimizers 
      │   
      ├── 📂 utils                                # Folder with utility functions
      ├── 📜 CelebAMask-HQ-attribute-anno.txt     # Matches between CelebA HQ images and attributes
      ├── 📜 available_directions.txt             # Info about available editings directions
      ├── 📜 requirements.txt                     # Lists required Python packages
      └── 📜 env_install.sh                       # Script to install necessary enviroment

