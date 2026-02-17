from utils_somecor import *

def plot_ecor_ncfile(ax,file_path,title):

    #-- open netcdf file
    ds = nc.Dataset(file_path)
    ecor_2D = ds.variables['ecoregion'][:]
    lats_2D = ds.variables['latitude'][:]
    lons_2D = ds.variables['longitude'][:]
    ds.close()
    print(f"{file_path} is loaded.")

    print(ecor_2D.min(),ecor_2D.max())

    #-- plot SOM classes on TP2 domain
    ecor_1D = ecor_2D.compressed()
    lons_1D = np.ma.array(lons_2D,mask=ecor_2D.mask).compressed() #-- apply land mask
    lats_1D = np.ma.array(lats_2D,mask=ecor_2D.mask).compressed() #-- apply land mask
    
    plot_ecoregion_TP2(ax,lons_1D,lats_1D,ecor_1D,title)

def plot_hisogram(ax,counts,title=None):
    # Plot histogram
    bars = ax.bar(range(1,len(counts)+1), counts, color='skyblue', edgecolor='black', label="Histogram")
    ax.set_xlabel("Region ID")
    ax.set_ylabel("No. of grids", color="black")
    ax.tick_params(axis='y', labelcolor="black")
    if title is not None:
        ax.set_title(title)

    # Display the grid for clarity
    ax.grid(True, linestyle="--", alpha=0.6)

def plot_histogram_w_cutoff(fig,ax1,percentile,sorted_counts,cumulative_percentage,cutoff_index,cutoff_value):
    # Plot sorted histogram
    bars = ax1.bar(range(1,len(sorted_counts)+1), sorted_counts, color='skyblue', edgecolor='black', label="Histogram")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("No. of grids", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.set_title("Histogram with Cumulative Percentage")

    # Plot the cumulative percentage on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(range(1,len(sorted_counts)+1), cumulative_percentage, color='orange', marker='o', markersize=2, label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")
    ax2.set_ylim(0, 100)

    # Highlight the cutoff value
    ax1.axvline(cutoff_index, color='red', linestyle='--', label=f"{percentile:02d}% Cutoff at index {cutoff_index}")
    ax1.text(cutoff_index, cutoff_value, f'  {cutoff_value}', color='red', fontsize=10, va='bottom')

    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.85), bbox_transform=ax1.transAxes)

    # Display the grid for clarity
    ax1.grid(True, linestyle="--", alpha=0.6)

def som_regionalization(task_control_dict,exp_config_dict,somclss_2D_dict):
    from skimage.measure import label
    from scipy import ndimage
    
    #--------------------------------
    # retrieve SOM input variables from a dictionary
    #--------------------------------
    isdebug       = task_control_dict['isdebug']
    isplot        = task_control_dict['isplot']
    isplot_map    = task_control_dict['isplot_map']
    isplot_hst    = task_control_dict['isplot_hst']
    isconfig_save = task_control_dict['isconfig_save']
    issomtrd_save = task_control_dict['issomtrd_save']
    issom_rnd     = task_control_dict['issom_rnd']
    issom_cls     = task_control_dict['issom_cls']
    issom_smth    = task_control_dict['issom_smth']
    isecor        = task_control_dict['isecor'] 

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
        
    #-----------------------------------------
    # Regionalization of SOM classes
    #-----------------------------------------

    #-- retrieve SOM classes from a dictionary

    clss_mskd_ma = somclss_2D_dict['clss_mskd_ma']
    lats_mskd_ma = somclss_2D_dict['lats_mskd_ma']
    lons_mskd_ma = somclss_2D_dict['lons_mskd_ma']
    dpth_mskd_ma = somclss_2D_dict['dpth_mskd_ma']

    #-- ecoregion index starts from 1 instead of 0
    clss_mskd_ma = clss_mskd_ma + 1

    #-- split geographically detached class regions
    land_mask = np.ma.getmaskarray(clss_mskd_ma)

    clss_mskd_ma.data[land_mask] = np.nan
    ecor_som = label(clss_mskd_ma, connectivity=1)
    ecor_som = np.ma.masked_array(ecor_som, mask=land_mask)
    ecor_som.data[land_mask] = 0

    unique_values,inverse_indices = np.unique(ecor_som.data, return_inverse=True)
    ecor_som_lbld = inverse_indices.reshape(ecor_som.shape)
    ecor_som_lbld = np.ma.masked_array(ecor_som_lbld, mask=land_mask)

    #-- Compute unique values and their counts
    # Exclude zeros (land mask) from the data before the counting
    filtered_data = ecor_som_lbld.data[ecor_som_lbld.data != 0]
    unique_values, counts = np.unique(filtered_data, return_counts=True)
    count_unique = len(unique_values)

    if isdebug:
        print(unique_values)
        print(count_unique)
        print(type(counts),counts)

        #-- plot SOM classes
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        plot_array(axes[0], clss_som      , "som class")
        plot_array(axes[1], clss_som_smthd, "som class smoothed")
        plt.tight_layout()
        plt.show()

        #-- plot histogram of unique values
        fig, ax1 = plt.subplots(figsize=(8, 3))
        plot_hisogram(ax1,counts,title=f"original counts")
        plt.tight_layout()
        plt.show()

    #-- Define theshold to select minor regions
    #percentile = 80 # [%]

    sorted_counts = np.sort(counts)[::-1]
    if isdebug:
        print(sorted_counts)

    # Compute cumulative sum and percentage
    cumulative_sum = np.cumsum(sorted_counts)
    cumulative_percentage = (cumulative_sum / cumulative_sum[-1]) * 100

    # Find the cutoff value index where cumulative percentage exceeds 95%
    cutoff_index = np.where(cumulative_percentage > percentile)[0][0]
    cutoff_value = sorted_counts[cutoff_index]

    # Print the cutoff value and index
    print(f"Cutoff index: {cutoff_index}, Cutoff value: {cutoff_value}")

    if isdebug:
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_histogram_w_cutoff(ax,sorted_counts,cumulative_percentage,cutoff_index, cutoff_value)
        plt.tight_layout()
        plt.show()

    #-- finalize threshold
    threshold = cutoff_value
    print(type(threshold),threshold)

    if isdebug:
        #-- list up major ecoregions
        print(f"------------------------")
        print(f" Major regions")
        print(f"------------------------")
        jdx = 0
        list_major = []
        for idx, cnt in enumerate(counts):
            if cnt > threshold:
                jdx = jdx + 1
                list_major.append(idx+1)
        print(list_major)
            
        #-- list up minor ecoregions
        print(f"------------------------")
        print(f" Minor regions")
        print(f"------------------------")
        jdx = 0
        list_minor = []
        for idx, cnt in enumerate(counts):
            if cnt <= threshold:
               jdx = jdx + 1
               list_minor.append(idx+1)
        print(list_minor)

    #-- merge minor region to dominant major region in its respective surroundings

    threshold = cutoff_value
    print(type(threshold),threshold)

    # Merge minor ecoregions to surrounding dominant major ecoregions

    array = ecor_som_lbld.data
    slices = ndimage.find_objects(array)
    for i,slc in enumerate(slices):
        padded_slice = tuple(
        slice(max(0, s.start - 1), min(array.shape[dim], s.stop + 1))
        for dim, s in enumerate(slc)
        )
        #segment = array[slc]
        segment = array[padded_slice]

        target_region = i + 1
        masked_segment = np.where(segment == target_region, np.nan, segment)
        surrounding_regions = np.unique(masked_segment[~np.isnan(masked_segment)]) # listup regions other than target region
        surrounding_regions = surrounding_regions.astype(int)                      # convert unique values to integers
        surrounding_regions = surrounding_regions[surrounding_regions != 0]        # ignore land_mask
    
        if counts[i] <= threshold:
            index_list = np.where(np.isin(unique_values, surrounding_regions))[0]
            values_at_indices = counts[index_list]
            max_index_in_list = index_list[np.argmax(values_at_indices)]
            if isdebug:
                print(f"Region {target_region} is surrounded by regions: {surrounding_regions}, The region with the largest count is: {unique_values[max_index_in_list]}")
            array[array==target_region] = unique_values[max_index_in_list]

    # Re-serialize the array
    unique_values,inverse_indices = np.unique(array,return_inverse=True)
    merged_array = inverse_indices.reshape(array.shape)

    if isdebug:
        inspect_array(ecor_som_lbld.data,True)
        inspect_array(merged_array,True)

    #-- Compute unique values and their counts
    # Exclude zeros (land mask) from the data before the counting
    filtered_data = merged_array[merged_array != 0]
    unique_values, merged_counts = np.unique(filtered_data, return_counts=True)
    count_unique = len(unique_values)

    if isdebug:
        print(unique_values)
        print(count_unique)
        print(type(merged_counts),merged_counts)

        #----------------------------------
        # plot histogram of unique values
        #----------------------------------
        fig, ax = plt.subplots(figsize=(8, 9))
        plot_hisogram(ax,merged_counts,title=f"merged counts")
        plt.tight_layout()
        plt.show()

    #-- plot summary of merging process

    if isplot_hst:
        fig, axes = plt.subplots(3,1,figsize=(7, 7.5))
        plot_hisogram(axes[0],counts,title=f"original counts")
        plot_histogram_w_cutoff(fig,axes[1],percentile,sorted_counts,cumulative_percentage,cutoff_index,cutoff_value)
        plot_hisogram(axes[2],merged_counts,title=f"merged counts")
        plt.tight_layout()
        plt.show()

    #-- finalize ecoregions
    ecor_da = merged_array.astype(np.float32)
    ecor_mskd_ma = np.ma.masked_array(ecor_da, mask=land_mask)

    if isdebug:
        print(ecor_mskd_ma.min(),ecor_mskd_ma.max())
    
    #-- register ecoregions on 2D grid to a dictionary
    ecor_2D_dict={
        'lats_mskd_ma': lats_mskd_ma,
        'lons_mskd_ma': lons_mskd_ma,
        'dpth_mskd_ma': dpth_mskd_ma,
        'ecor_mskd_ma': ecor_mskd_ma,
    }

    return ecor_2D_dict