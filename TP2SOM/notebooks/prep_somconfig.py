from utils_somecor import *

def load_somconfig(root_io_dict, trial):
    #------------------------------------
    # load the SOM configuration to yaml file (optional)
    #------------------------------------
    ROOT_SAVE_DIR = root_io_dict['ROOT_SAVE_DIR']
    topology, percentile = "5x3", 90
    PATH_YAML = f"{ROOT_SAVE_DIR}{topology}/{percentile:02d}/{trial:02d}/"
    yaml_filename = f"{PATH_YAML}SOM_config.yaml"
    
    try:
        with open(yaml_filename, 'r') as yaml_file:
            som_config_dict = yaml.safe_load(yaml_file)
            print(f"SOM config file: {yaml_filename} loaded")
    except FileNotFoundError:
        print(f"Error: The file '{yaml_filename}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        
    return som_config_dict

def save_somconfig(task_control_dict, root_io_dict, setup_experiment, topology, percentile, rseed, trial=None):
    #------------------------------------
    # save experiment configurations to yaml file
    #------------------------------------
    if trial is None:
        som_config_dict = setup_experiment(task_control_dict, root_io_dict, topology, percentile, rseed)
    else:
        som_config_dict = setup_experiment(task_control_dict, root_io_dict, topology, percentile, rseed, trial)
    PATH_YAML = som_config_dict['PATH_DATA']
    yaml_filename = f"{PATH_YAML}SOM_config.yaml"
    
    try:
        directory_path = Path(PATH_YAML)
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directories '{directory_path}' created")
    except OSError as error:
        print(f"Error creating directories '{directory_path}': {error}")
        
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(som_config_dict, yaml_file, default_flow_style=False)
        print(f"SOM config file: {yaml_filename} saved")
        
    return som_config_dict