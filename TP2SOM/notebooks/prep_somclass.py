from utils_somecor import *

def plot_som_dict(ax,somdict,title):
    #-- retrieve 2D SOM data from a dictionary
    lats_mskd_ma = somdict['lats_mskd_ma']
    lons_mskd_ma = somdict['lons_mskd_ma']
    clss_mskd_ma = somdict['clss_mskd_ma']
    
    #-- flatten data
    lats_som_1D = lats_mskd_ma.compressed()
    lons_som_1D = lons_mskd_ma.compressed()
    clss_som_1D = clss_mskd_ma.compressed() + 1 #-- shift som class index by 1 for plotting
    
    plot_cluster_TP2(ax,lons_som_1D,lats_som_1D,clss_som_1D,title)

def plot_som_ncfile(ax,file_path,title):

    #-- open netcdf file
    ds = nc.Dataset(file_path)
    clss_som = ds.variables['class'][:]
    lats_som = ds.variables['latitude'][:]
    lons_som = ds.variables['longitude'][:]
    ds.close()
    print(f"{file_path} is loaded.")

    #-- plot SOM classes on TP2 domain
    clss_som_1D = clss_som.compressed() + 1                             #-- shift som class index by 1 for plotting
    lons_som_1D = np.ma.array(lons_som,mask=clss_som.mask).compressed() #-- apply land mask
    lats_som_1D = np.ma.array(lats_som,mask=clss_som.mask).compressed() #-- apply land mask
    
    plot_cluster_TP2(ax,lons_som_1D,lats_som_1D,clss_som_1D,title)

def save_som_dict_to_ncfile(somdict,exp_config_dict,vardict='clss_mskd_ma',varname='class',file_nc='TP2_clss.nc'):
    #---------------------------------------
    # save SOM classes or ecoregions (starting from 0) to a 2D netcdf file
    #---------------------------------------
    PATH_DATA = exp_config_dict['PATH_DATA']
    file_path=f"{PATH_DATA}{file_nc}"

    #-- remove old file
    if os.path.exists(file_path):
        os.remove(file_path)
    
    #-- retrieve 2D SOM data from a dictionary
    lats_mskd_ma = somdict['lats_mskd_ma']
    lons_mskd_ma = somdict['lons_mskd_ma']
    dpth_mskd_ma = somdict['dpth_mskd_ma']
    vars_mskd_ma = somdict[vardict]
    
    jdim = vars_mskd_ma.shape[0]
    idim = vars_mskd_ma.shape[1]
    
    # Create a new NetCDF file
    dataset = Dataset(file_path, 'w', format='NETCDF4_CLASSIC')

    # Create dimensions
    dataset.createDimension('jdim',jdim)
    dataset.createDimension('idim',idim)

    # Create variables
    latitudes = dataset.createVariable('latitude', np.float32, ('jdim','idim'))
    longitudes = dataset.createVariable('longitude', np.float32, ('jdim','idim'))
    depths = dataset.createVariable('depth', np.float32, ('jdim','idim'), fill_value=1e+20)
    variables = dataset.createVariable(varname, np.float32, ('jdim','idim'), fill_value=1e+20)

    # Assign attributes
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    depths.units = 'm'
    variables.units = 'None'

    # Write data to variables
    latitudes[:,:]  = lats_mskd_ma.data
    longitudes[:,:] = lons_mskd_ma.data
    depths[:,:]     = dpth_mskd_ma.filled(1e+20)
    variables[:,:]  = vars_mskd_ma.filled(1e+20)  # Use .filled() method to replace masked values

    # Close the dataset
    dataset.close()

    print(f"{file_path} is saved.")

def load_somclss_2D(list_io_dict, file_somclss=None):
    PATH_DATA  = list_io_dict['PATH_DATA']
    
    if file_trdat is None:
        file_trdat = f"somclss_2D.pkl"
        
    file_path = f"{PATH_DATA}{file_somclss}"

    with open(file_path, 'rb') as file:
        somclss_2D_dict = pickle.load(file)
        print(f"SOM classes data file: '{file_path}' loaded")

    return traindata_dict

def som_clssification_2D(task_control_dict,traindata_dict,exp_config_dict,file_somclss=None):

    if file_somclss is None:
        file_somclss = "somclss_2D.pkl"
    
    #--------------------------------
    # retrieve SOM input variables from a dictionary
    #--------------------------------
    isdebug       = task_control_dict['isdebug']
    isplot        = task_control_dict['isplot']
    isplot_map    = task_control_dict['isplot_map']
    isconfig_save = task_control_dict['isconfig_save']
    issomtrd_save = task_control_dict['issomtrd_save']
    issom_rnd     = task_control_dict['issom_rnd']
    issom_cls     = task_control_dict['issom_cls']
    issom_smth    = task_control_dict['issom_smth']
    isecor        = task_control_dict['isecor'] 
    
    #--------------------------------
    # retrieve SOM input variables from a dictionary
    #--------------------------------
    if isdebug:
        inspect_dictionary(traindata_dict)
        
    arrays_norm_dict = traindata_dict['arrays_norm_dict']
    if isdebug:
        inspect_dictionary(arrays_norm_dict)
    lats_mskd_ma     = traindata_dict['lats_mskd_ma']
    lons_mskd_ma     = traindata_dict['lons_mskd_ma']
    dpth_mskd_ma     = traindata_dict['dpth_mskd_ma']
    land_mask        = traindata_dict['land_mask']

    #--------------------------------
    # retrieve SOM input settings from a dictionary
    #--------------------------------
    if isdebug:
        inspect_dictionary(exp_config_dict)
    
    trial                 = exp_config_dict['trial']
    rseed                 = exp_config_dict['rseed']
    mdim                  = exp_config_dict['mdim']
    ndim                  = exp_config_dict['ndim']
    nclss                 = exp_config_dict['nclss']
    pdim                  = exp_config_dict['pdim']
    topology              = exp_config_dict['topology']
    percentile            = exp_config_dict['percentile']
    name_config_list      = exp_config_dict['name_config_list']
    name_parameters_list  = exp_config_dict['name_parameters_list']
    name_variables_list   = exp_config_dict['name_variables_list']
    isuse_config_list     = exp_config_dict['isuse_config_list']
    isuse_parameters_list = exp_config_dict['isuse_parameters_list']
    isuse_variables_list  = exp_config_dict['isuse_variables_list']
    PATH_DATA             = exp_config_dict['PATH_DATA']
    PATH_FIGS             = exp_config_dict['PATH_FIGS']
    
    #--------------------------------
    # setup input parameters for SOM
    #--------------------------------
    print(f"----------------------------------")
    print(f" Concatenate SOM inputs")
    print(f"----------------------------------")

    # concatenate numpy.ndarray in 1D array after removing land

    alist = []

    # configuration
    
    for name, isuse in zip(name_config_list, isuse_config_list):
        if isuse:
            print(name)
            if isdebug:
                inspect_array(arrays_norm_dict[name],isplot)
            alist.append(arrays_norm_dict[name].compressed()) # compress 2D > 1D (remove land)

    # parameters

    for name, isuse in zip(name_parameters_list, isuse_parameters_list):
        if isuse:
            print(name)
            if isdebug:
                inspect_array(arrays_norm_dict[name],isplot)
            alist.append(arrays_norm_dict[name].compressed())  # compress 2D > 1D (remove land)

    # annual mean
    
    for name, isuse in zip(name_variables_list, isuse_variables_list):
        if isuse:
            print('ann_'+name)
            if isdebug:
                inspect_array(arrays_norm_dict['ann_'+name],isplot)
            alist.append(arrays_norm_dict['ann_'+name].compressed())  # compress 2D > 1D (remove land)

     # seasonal change
    
    for name, isuse in zip(name_variables_list, isuse_variables_list):
        if isuse:
            print('dlt_'+name)
            if isdebug:
                inspect_array(arrays_norm_dict['dlt_'+name],isplot)
            alist.append(arrays_norm_dict['dlt_'+name].compressed())  # compress 2D > 1D (remove land)
                
    # flatten data
    
    data_som = np.stack(alist,axis=1)
    lats_som = lats_mskd_ma.compressed()
    lons_som = lons_mskd_ma.compressed()

    if isdebug:
        inspect_array(data_som,False)
        inspect_array(lons_som,False)
        inspect_array(lats_som,False)

    #---------------------------------------
    # Randamize input data. We use 100% of trainig data so far (frac=1.0).
    # Random sampling of data_som along axis=0
    #---------------------------------------
    
    if issom_rnd:
        #-- random sampling of input data for training        
        frac = 1.0 # fraction of data_som to be sampled [0-1]
        dlen = data_som.shape[0]
        random.seed(10)
        inds = random.sample( list(range(dlen)), int(frac*dlen) )
    
        if isdebug:
            print('INDS:',inds[:10])

        data_train = data_som[inds,:]
        lons_train = lons_som[inds]
        lats_train = lats_som[inds]

        if isdebug:
            inspect_array(data_train,False)
            inspect_array(lons_train,isplot)
            inspect_array(lats_train,isplot)
        
    #---------------------------------------
    # Perform SOM classification
    #---------------------------------------

    print(f"----------------------------------")
    print(f" Perform SOM classification")
    print(f"----------------------------------")
    print(f"Trial ID            : {trial:02d}")
    print(f"SOM topology        : {topology}")
    print(f"Number of parameters: {pdim}")

    #-- train SOM classes
    train_som = SOM(m=mdim,n=ndim,dim=pdim,random_state=rseed)
    train_som.fit(data_train)

    #-- assign SOM classes to given data        
    clss_som = train_som.predict(data_som) 

    if isdebug:
        print(f"clss_som shape:{clss_som.shape}")

    #---------------------------------------
    # remap SOM classes to original 2D grid
    #---------------------------------------

    original_shape = lats_mskd_ma.shape
    original_dtype = lats_mskd_ma.dtype
    if isdebug:
        print(original_shape)
        print(original_dtype)
        print(land_mask.shape)
        print((~land_mask).sum())

    clss_da = np.empty(original_shape,dtype=original_dtype)
    clss_da[~land_mask] = clss_som
    clss_mskd_ma = np.ma.array(clss_da, mask=land_mask)

    if isdebug:
        inspect_array(clss_mskd_ma,isplot)

    #---------------------------------------
    # register SOM classes on 2D grid to a dictionary
    #---------------------------------------
    somclss_2D_dict={
        'lats_mskd_ma': lats_mskd_ma,
        'lons_mskd_ma': lons_mskd_ma,
        'dpth_mskd_ma': dpth_mskd_ma,
        'clss_mskd_ma': clss_mskd_ma,
    }

    #---------------------------------------
    # apply 2D smoothing on SOM classes on 2D grid
    #---------------------------------------
    if issom_smth:
        #-- Define the number of smoothing iterations                                                                                                     
        num_iterations = 30
        
        print(f"----------------------------------")
        print(f" Apply 2D smoothing to SOM classes on 2D grid")
        print(f"----------------------------------")
        print(f"Number of iterations: {num_iterations}")

        clss = clss_mskd_ma.filled(1e+20)
        clss_smthd = apply_smoothing_preserve_land(clss, land_mask, num_iterations)
        clss_smthd_mskd_ma = np.ma.array(clss_smthd, mask=land_mask)

        if isdebug:
            inspect_array(land_mask,True)
            inspect_array(lats_mskd_ma,True)
            inspect_array(lons_mskd_ma,True)
            isdebug=True
            inspect_array(clss_mskd_ma,True)
            inspect_array(clss_smthd_mskd_ma,True)
            isdebug=False

        #---------------------------------------
        # register SOM classes on 2D grid to a dictionary
        #---------------------------------------
        somclss_2D_dict={
            'lats_mskd_ma': lats_mskd_ma,
            'lons_mskd_ma': lons_mskd_ma,
            'dpth_mskd_ma': dpth_mskd_ma,
            'clss_mskd_ma': clss_smthd_mskd_ma,
        }

        #---------------------------------------
        # save a dictionary
        #---------------------------------------
        directory_path = Path(PATH_DATA)
        file_path = f"{PATH_DATA}{file_somclss}"

        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directories '{directory_path}' created")
        except OSError as error:
            print(f"Error creating directories '{directory_path}': {error}")

        with open(file_path, 'wb') as file:
            pickle.dump(traindata_dict, file)
            print(f"SOM classes data file: '{file_path}' saved")
    
    return somclss_2D_dict