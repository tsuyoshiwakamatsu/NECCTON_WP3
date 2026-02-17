from utils_somecor import *

def load_traindata(list_io_dict, file_trdat=None, isdebug=None):
    PATH_LOAD_DIR  = list_io_dict['ROOT_SAVE_DIR']
    
    if isdebug is None:
        isdebug = False

    if file_trdat is None:
        file_trdat = f"som_traindata.pkl"
        
    file_path = f"{PATH_LOAD_DIR}{file_trdat}"

    with open(file_path, 'rb') as file:
        traindata_dict = pickle.load(file)
        print(f"SOM training data file: '{file_path}' loaded")

    return traindata_dict

def save_traindata(list_io_dict, som_config_dict, file_trdat=None, isdebug=None):
    PATH_INPUT_DIR = list_io_dict['ROOT_LOAD_DIR']
    PATH_SAVE_DIR  = list_io_dict['ROOT_SAVE_DIR']
    
    name_config_list     = som_config_dict['name_config_list']
    name_parameters_list = som_config_dict['name_parameters_list']
    name_variables_list  = som_config_dict['name_variables_list']
    
    if isdebug is None:
        isdebug = False

    if file_trdat is None:
        file_trdat = f"som_traindata.pkl"
        
    file_path = f"{PATH_SAVE_DIR}{file_trdat}"
    
    #--------------------------------
    # hycom input file
    #--------------------------------
    varfile = f"{PATH_INPUT_DIR}daily_surface_climatology_2007_2016.nc"

    #--------------------------------
    # Gaussian filter settings
    #--------------------------------
    kernel_size=5
    sigma=2.0
    sigma_depth=2.0
    sigma_chlmaxday=2.0
    iterations=5
    iterations_depth=10
    iterations_chlmaxday=20

    #--------------------------------
    # preprocess hycom model configurations
    #--------------------------------

    with NetCDFHandler(varfile) as ncfile:
        fill_value = ncfile.variables['depth'].getncattr('_FillValue')
        print('FillValue',fill_value)
        dpth = ncfile.variables['depth'][:,:]
        dpth_da = ncfile.variables['depth'][:,:].data
        lons_da = ncfile.variables['longitude'][:,:].data
        lats_da = ncfile.variables['latitude'][:,:].data
        
        #--------------------------------
        # define land mask
        #--------------------------------
        land_mask = np.ones(dpth.shape, dtype=bool)
        land_mask[dpth_da != fill_value] = False
        print('land_mask', np.sum(land_mask))

        #--------------------------------                                                                                                       
        # update land mask                                                                                                                      
        #--------------------------------                                                                                                       

        fill_value = ncfile.variables['temp'].getncattr('_FillValue')
        print('FillValue',fill_value)

        temp2D_da = ncfile.variables['temp'][0,:,:,:].data.squeeze()
        temp_mask = np.zeros(temp2D_da.shape, dtype=bool)
        temp_mask[temp2D_da <= fill_value] = True

        land_mask[temp_mask == 1] = 1
        print('land_mask', np.sum(land_mask))

        #--------------------------------
        # apply Gaussian filter on depth data
        #--------------------------------
    
        dpth_smthd = iterate_convolution_2D(dpth,kernel_size,sigma_depth,iterations_depth)
        dpth_da = dpth_smthd.data
    
        if isdebug:
            inspect_array(dpth,True)
            inspect_array(dpth_smthd,True)
            
        #--------------------------------
        # convert longitude to point on circle
        #--------------------------------
        lons_rad_da = np.radians(lons_da) # degree to radian
        # Calculate x and y coordinates on the unit circle
        lonx_da = np.cos(lons_rad_da)
        lony_da = np.sin(lons_rad_da)

        if isdebug:
            inspect_array(lonx_da,isplot)
            inspect_array(lony_da,isplot)
            
        #--------------------------------
        # apply common land mask to config variables
        #--------------------------------

        dpth_mskd_ma = ma.masked_array(dpth_da,mask=land_mask)
        lats_mskd_ma = ma.masked_array(lats_da,mask=land_mask)
        lons_mskd_ma = ma.masked_array(lons_da,mask=land_mask)
        lonx_mskd_ma = ma.masked_array(lonx_da,mask=land_mask)
        lony_mskd_ma = ma.masked_array(lony_da,mask=land_mask)

        if isdebug:
            inspect_array(dpth_mskd_ma,isplot)
            inspect_array(lats_mskd_ma,isplot)
            inspect_array(lons_mskd_ma,isplot)
            inspect_array(lonx_mskd_ma,isplot)
            inspect_array(lony_mskd_ma,isplot)

        #--------------------------------
        # normalize config variables
        #--------------------------------

        dpth_mskd_norm_ma = normalize_array(dpth_mskd_ma)
        lats_mskd_norm_ma = normalize_array(lats_mskd_ma)
        lons_mskd_norm_ma = normalize_array(lons_mskd_ma)
        lonx_mskd_norm_ma = normalize_array(lonx_mskd_ma)
        lony_mskd_norm_ma = normalize_array(lony_mskd_ma)

        if isdebug:
            inspect_array(dpth_mskd_norm_ma,isplot)
            inspect_array(lats_mskd_norm_ma,isplot)
            inspect_array(lons_mskd_norm_ma,isplot)
            inspect_array(lonx_mskd_norm_ma,isplot)
            inspect_array(lony_mskd_norm_ma,isplot)

        #--------------------------------
        # register configurations to dictionary
        #--------------------------------
    
        arrays_dict = {}
        arrays_norm_dict = {}

        # masked array
    
        arrays_dict['dpth'] = dpth_mskd_ma 
        arrays_dict['lats'] = lats_mskd_ma 
        arrays_dict['lons'] = lons_mskd_ma 
        arrays_dict['lonx'] = lonx_mskd_ma 
        arrays_dict['lony'] = lony_mskd_ma

        if isdebug:
            inspect_dictionary(arrays_dict)
            inspect_array(arrays_dict['dpth'],isplot)
            inspect_array(arrays_dict['lats'],isplot)
            inspect_array(arrays_dict['lons'],isplot)
            inspect_array(arrays_dict['lonx'],isplot)
            inspect_array(arrays_dict['lony'],isplot)

        # normalized array
    
        arrays_norm_dict['dpth'] = dpth_mskd_norm_ma 
        arrays_norm_dict['lats'] = lats_mskd_norm_ma 
        arrays_norm_dict['lons'] = lons_mskd_norm_ma 
        arrays_norm_dict['lonx'] = lonx_mskd_norm_ma 
        arrays_norm_dict['lony'] = lony_mskd_norm_ma
        
        inspect_dictionary(arrays_norm_dict)
    
        if isdebug:
            inspect_dictionary(arrays_norm_dict)
            inspect_array(arrays_norm_dict['dpth'],isplot)
            inspect_array(arrays_norm_dict['lats'],isplot)
            inspect_array(arrays_norm_dict['lons'],isplot)
            inspect_array(arrays_norm_dict['lonx'],isplot)
            inspect_array(arrays_norm_dict['lony'],isplot)

    #--------------------------------
    # preprocess hycom model states
    #--------------------------------

    # define seasons
        
    year = 2007 # any noleap year
    months_jfm = [1,2,3]
    months_jas = [7,8,9]
                
    with NetCDFHandler(varfile) as ncfile:
        # You can now access the data and metadata in the NetCDF file
        print(ncfile.variables.keys())

        # preprocess parameters

        for name in name_parameters_list:
            print(name)
            if name == 'chlmaxday':
                vars_tmp = (ncfile.variables['ECO_diac'][:,:,:].data +
                            ncfile.variables['ECO_flac'][:,:,:].data +
                            ncfile.variables['ECO_cclc'][:,:,:].data)
                vars_tmp = vars_tmp.squeeze()
                
                # adjust the maximum chl-a date by setting the time e.g. between Mar and Oct 
                vars_tmp = np.argmax(vars_tmp[60:270], axis=0)
                vars_tmp_mskd_ma = ma.masked_array(vars_tmp,mask=land_mask)

                #-- iteration of Gaussian filter
                vars_smthd = iterate_convolution_2D(vars_tmp_mskd_ma,kernel_size,sigma_chlmaxday,iterations_chlmaxday)
                vars_smthd_mskd_ma = ma.masked_array(vars_smthd.data,mask=land_mask)
                inspect_array(vars_smthd_mskd_ma,True)
            
            else:
                vars_tmp = ncfile.variables[name][:,:,:].data
                vars_tmp = vars_tmp.squeeze()
                vars_tmp_mskd_ma = ma.masked_array(vars_tmp,mask=land_mask)
                
                #-- iteration of Gaussian filter
                vars_smthd = iterate_convolution_2D(vars_tmp_mskd_ma,kernel_size,sigma,iterations)
                vars_smthd_mskd_ma = ma.masked_array(vars_smthd.data,mask=land_mask)

            #-- normalization
            vars_smthd_mskd_norm_ma = normalize_array(vars_smthd_mskd_ma)

            #-- register to the dictionary
            arrays_dict[name] = vars_smthd_mskd_ma
            arrays_norm_dict[name] = vars_smthd_mskd_norm_ma

        inspect_dictionary(arrays_norm_dict)

        # preprocess variables
        
        for name in name_variables_list:
            print(name)
            if name == 'chla':
                vars_da = (ncfile.variables['ECO_diac'][:,:,:].data +
                           ncfile.variables['ECO_flac'][:,:,:].data +
                           ncfile.variables['ECO_cclc'][:,:,:].data)
            else:
                vars_da = ncfile.variables[name][:,:,:].data

            vars_da = vars_da.squeeze()
                    
            if isdebug:
                inspect_array(vars_da,isplot)

            #-- annual mean and seasonal mean

            vars_ann_da = vars_da.mean(axis=0)
            vars_ann_mskd_ma = ma.masked_array(vars_ann_da,mask=land_mask)

            if isdebug:
                inspect_array(vars_ann_da,isplot)
                inspect_array(vars_ann_mskd_ma,isplot)
                    
            #-- winter (JFM) mean
                    
            vars_list = [extract_month_data(vars_da, month, year) for month in months_jfm]
            vars_jfm_da = np.concatenate(vars_list,axis=0)
            vars_jfm_da = vars_jfm_da.mean(axis=0)

            #-- summer (JAS) mean

            vars_list = [extract_month_data(vars_da, month, year) for month in months_jas]
            vars_jas_da = np.concatenate(vars_list,axis=0)
            vars_jas_da = vars_jas_da.mean(axis=0)
                    
            if isdebug:
                inspect_array(vars_jfm_da,isplot)
                inspect_array(vars_jas_da,isplot)
                    
            #-- delta: seasonal contrast (summer - winter)
                    
            vars_dlt_da = vars_jas_da - vars_jfm_da
            vars_dlt_mskd_ma = ma.masked_array(vars_dlt_da,mask=land_mask)

            if isdebug:
                inspect_array(vars_dlt_da,isplot)
                inspect_array(vars_dlt_mskd_ma,isplot)

            #-- iteration of Gaussian filter

            if isdebug:
                inspect_array(vars_ann_mskd_ma,True)

            vars_smthd = iterate_convolution_2D(vars_ann_mskd_ma,kernel_size,sigma,iterations)
            vars_ann_mskd_ma = ma.masked_array(vars_smthd.data,mask=land_mask)

            if isdebug:
                inspect_array(vars_ann_mskd_ma,True)               
                inspect_array(vars_dlt_mskd_ma,True)

            vars_smthd = iterate_convolution_2D(vars_dlt_mskd_ma,kernel_size,sigma,iterations)
            vars_dlt_mskd_ma = ma.masked_array(vars_smthd.data,mask=land_mask)
            
            if isdebug:
                inspect_array(vars_dlt_mskd_ma,True)               
                
            #-- normalization

            vars_ann_mskd_norm_ma = normalize_array(vars_ann_mskd_ma)
            vars_dlt_mskd_norm_ma = normalize_array(vars_dlt_mskd_ma)
            
            if isdebug:
                inspect_array(vars_ann_mskd_norm_ma,isplot)
                inspect_array(vars_dlt_mskd_norm_ma,isplot)
                    
            #-- register to the dictionary
                    
            # masked array
            arrays_dict['ann_'+name] = vars_ann_mskd_ma
            arrays_dict['dlt_'+name] = vars_dlt_mskd_ma

            # normalized array
            arrays_norm_dict['ann_'+name] = vars_ann_mskd_norm_ma
            arrays_norm_dict['dlt_'+name] = vars_dlt_mskd_norm_ma

    if isdebug:
        inspect_dictionary(arrays_dict)
        inspect_dictionary(arrays_norm_dict)

    #---------------------------------------
    # save som training data to pickle file
    #---------------------------------------

    traindata_dict = {}

    traindata_dict['arrays_dict'] = arrays_dict
    traindata_dict['arrays_norm_dict'] = arrays_norm_dict
    traindata_dict['lats_mskd_ma'] = lats_mskd_ma
    traindata_dict['lons_mskd_ma'] = lons_mskd_ma
    traindata_dict['dpth_mskd_ma'] = dpth_mskd_ma
    traindata_dict['land_mask'] = land_mask
    
    directory_path = Path(PATH_SAVE_DIR)

    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directories '{directory_path}' created")
    except OSError as error:
        print(f"Error creating directories '{directory_path}': {error}")

    with open(file_path, 'wb') as file:
        pickle.dump(traindata_dict, file)
        print(f"SOM training data file: '{file_path}' saved")

    return traindata_dict