import scipy.stats as stats
import scipy as sp 
import time as time
from sklearn.mixture import GaussianMixture
import os
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.transform import from_origin
from pyproj import Proj, Transformer
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
from scipy.interpolate import interp1d
import warnings
import time 
import gc
from scipy.signal import lombscargle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from skimage.measure import block_reduce
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from skimage.measure import label
from skimage.measure import regionprops
#own functions
from functions_ccdc import *



# a funciton for preparing image data for viewing
def stretchIm(Im, bands, l, h):
    """
    Applies percentile contrast stretch to an image, ignoring black pixels (all zeros).
    
    Parameters:
    - Im (numpy array): 3D image array (rows x cols x bands).
    - bands (list or array): List of band indices to apply stretching.
    - l (float): Low percentile (e.g., 2 for 2% stretch).
    - h (float): High percentile (e.g., 98 for 98% stretch).
    
    Returns:
    - ImSub (numpy array): Stretched image with values scaled to [0,1].
    """

    # Extract selected bands and make a copy to avoid modifying original data
    ImSub = Im[:, :, bands].copy()

    # Apply stretching to each band separately
    for i in range(ImSub.shape[2]):  # Loop over selected bands
        # Compute percentiles only on non-zero pixels
        low_percentile, high_percentile = np.percentile(ImSub[ImSub > 0], [l, h])

        # Prevent division by zero
        range_adjusted = np.where(high_percentile - low_percentile == 0, 1, high_percentile - low_percentile)

        # Apply contrast stretching
        ImSub[:, :, i] = (ImSub[:, :, i] - low_percentile) / range_adjusted

    # Clip values to [0,1] range
    ImSub = np.clip(ImSub, 0, 1)

    return ImSub 


#single date then looking over it
def extract_single_date(name, slash=True):
    """
    Extracts date and time from a filename formatted as YYYYMMDD_HHMMSS.
    
    Parameters:
    - name (str): Filename containing a date in the format 'YYYYMMDD_HHMMSS'.
    - slash (bool): Whether the filename contains directory paths separated by '/'.

    Returns:
    - date (numpy array): 7-element array containing [Year, Month, Day, Hour, Minute, Second, Fractional Year].
    """
    
    try:
        # Extract filename without path if slash=True
        if slash:
            name = name.split('/')[-1]

        # Split by underscore to separate date and time
        parts = name.split('_')
        if len(parts) < 2:
            raise ValueError(f"Invalid filename format: {name}")

        date = np.zeros(7)

        # Extract date components
        date[0] = float(parts[0][0:4])  # Year
        date[1] = float(parts[0][4:6])  # Month
        date[2] = float(parts[0][6:8])  # Day

        # Extract time components
        date[3] = float(parts[1][0:2])  # Hour
        date[4] = float(parts[1][2:4])  # Minute
        date[5] = float(parts[1][4:6])  # Second

        # Month-day mapping (default to non-leap year)
        mvec = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Adjust for leap years (century rule: must be divisible by 400)
        year_int = int(date[0])
        if (year_int % 4 == 0 and year_int % 100 != 0) or (year_int % 400 == 0):
            mvec[1] = 29  # Leap year February has 29 days

        # Compute fractional year representation
        days_in_year = sum(mvec)
        days_before_month = sum(mvec[:int(date[1]) - 1])  # Days in previous months
        date[6] = date[0] + (days_before_month + (date[2] - 1)) / days_in_year + \
                  date[3] / (days_in_year * 24) + date[4] / (days_in_year * 24 * 60) + \
                  date[5] / (days_in_year * 24 * 3600)

        return date

    except Exception as e:
        raise ValueError(f"Error processing filename '{name}': {e}")


def loop_dates(list_names, slash=False):
    """
    Extracts date and time information from a list of filenames.

    Parameters:
    - list_names (list of str): List of filenames containing date-time information.
    - slash (bool): Whether filenames contain directory paths (default: False).

    Returns:
    - result (numpy array): (7 x N) array with extracted date components:
        [Year, Month, Day, Hour, Minute, Second, Fractional Year].
    """
    
    dates = []  # Store extracted date arrays
    
    for i, name in enumerate(list_names):
        try:
            date_info = extract_single_date(name, slash=slash)

            # Ensure the extracted date has 7 elements
            if len(date_info) != 7:
                raise ValueError(f"Extracted date from '{name}' has incorrect length: {len(date_info)}")

            dates.append(date_info)

        except Exception as e:
            print(f"Warning: Skipping '{name}' due to error: {e}")

    # Convert to NumPy array if at least one valid entry exists
    if len(dates) > 0:
        result = np.vstack(dates).T  # Transpose to match (7 x N) format
    else:
        result = np.zeros((7, 0))  # Return an empty (7x0) array if no valid data
    
    return result  

#not used currently, but could be useful
def get_snow(data,lim_1_7=0.65,lim_2_8=0.5):
    meds=np.median(data,axis=2)
    rat17=meds[0,:]/meds[6,:]
    rat28=meds[1,:]/meds[7,:]  
    snow=np.where((rat17>lim_1_7) & (rat28>lim_2_8),1,0)
    return snow

#not used currently, but could be useful
def process_snow(data,dates,lim_1_7=0.65,lim_2_8=0.5):
    snow=get_snow(data,lim_1_7=lim_1_7,lim_2_8=lim_2_8)
    ndata=data[:,(snow==0),:]
    ndates=dates[:,(snow==0)]
    return ndata,ndates


#getting the tif files used, and get the dates when they were observed 

def get_files_and_dates(dir_list, before=None, appended='', slash=True, meta=False):
    """
    Searches for specific TIFF files in given directories, sorts them by name, 
    and extracts date information.

    Parameters:
    - dir_list (list of str): List of subdirectories to search.
    - before (str or None): Base directory path (default: current working directory).
    - appended (str): Additional string to append to each directory name.
    - slash (bool): Whether filenames contain directory paths (default: True).
    - meta (bool): If True, searches for 'udm2_clip.tif'; otherwise, searches for 
                    '3B_AnalyticMS_SR_8b_clip.tif'.

    Returns:
    - res (numpy array): 7xN matrix of extracted dates.
    - dir_fnames_sorted (list of str): Sorted full file paths.
    - fnames_sorted (list of str): Sorted file names.
    """

    # Set base directory to current directory if not provided
    if before is None:
        before = os.getcwd()

    # Construct full directory paths
    directories = [os.path.join(before, d + appended) for d in dir_list]
    print(f"Searching in directories: {directories}")

    # File search pattern
    search_pattern = 'udm2_clip.tif' if meta else '3B_AnalyticMS_SR_8b_clip.tif'

    # Initialize lists to store filenames and full paths
    fnames = []
    dir_fnames = []

    # Search for files
    for directory in directories:
        for subdir, _, files in os.walk(directory):
            for fname in files:
                if search_pattern in fname:
                    fnames.append(fname)
                    dir_fnames.append(os.path.join(subdir, fname))

    print(f"Found {len(fnames)} matching files.")

    # Sort filenames and associated paths
    sorted_files = sorted(zip(fnames, dir_fnames), key=lambda x: x[0])
    fnames_sorted, dir_fnames_sorted = zip(*sorted_files) if sorted_files else ([], [])

    print(f"Sorted filenames: {fnames_sorted}")

    # Extract dates
    res = loop_dates(list(fnames_sorted), slash=slash)

    return res, list(dir_fnames_sorted), list(fnames_sorted)


#function to convert data to numpy cube
def tiff_to_cube(file, x1=0, x2=None, y1=0, y2=None):
    """
    Reads a TIFF file and converts it into a 3D NumPy array (rows × cols × bands).
    
    Parameters:
    - file (str): Path to the TIFF file.
    - x1, x2 (int): Row index range to extract (default: full image).
    - y1, y2 (int): Column index range to extract (default: full image).

    Returns:
    - image_array (numpy array): 3D array (rows × cols × bands).
    """

    # Open the TIFF file using a context manager
    with rio.open(file) as image:
        # Validate file format
        if image.driver != "GTiff":
            raise ValueError(f"Error: '{file}' is not a valid GeoTIFF file.")

        # Get image dimensions
        nBands = image.count
        nRows, nCols = image.height, image.width

        # Read all bands as a NumPy array and transpose to (rows, cols, bands)
        image_array = image.read().transpose((1, 2, 0))  # (bands, rows, cols) → (rows, cols, bands)

        # Ensure proper slicing (handle None or -1 cases)
        x2 = nRows if x2 is None or x2 == -1 else x2
        y2 = nCols if y2 is None or y2 == -1 else y2

        return image_array[x1:x2, y1:y2, :]

#preprocesses one image and meta image  # it should be added still  getting also the meta data 

def preprocess_privthi(image, qual, select='rgbn', downsample=True, samp_fact=10):
    """
    Preprocesses a remote sensing image by selecting specific spectral bands and downsampling.

    Parameters:
    - image (numpy array): 3D array (rows × cols × bands) containing spectral data.
    - qual (numpy array): 3D array (rows × cols × 8) containing quality flags.
    - select (str): Band selection ('rgbn' for red, green, blue, NIR or 'all' for all bands).
    - downsample (bool): Whether to downsample the image.
    - samp_fact (int): Downsampling factor.

    Returns:
    - image_new (numpy array): Downsampled spectral image.
    - qual_new (numpy array): Downsampled quality data.
    """

    # Define spectral band indices
    if select == 'rgbn':
        chan = [1, 3, 5, 7]  # RGB + NIR
    elif select == 'all':
        chan = np.arange(8)  # All 8 bands
    else:
        raise ValueError("Invalid selection. Choose 'rgbn' or 'all'.")

    # Select channels
    image_sel = image if select == 'all' else image[:, :, chan]

    # Downsampling
    if downsample:
        # Ensure valid downsampling region
        rows, cols, _ = image.shape
        new_rows = rows - (rows % samp_fact)
        new_cols = cols - (cols % samp_fact)

        # Initialize downsampled arrays
        image_new = np.zeros((new_rows // samp_fact, new_cols // samp_fact, len(chan)))
        qual_new = np.zeros((new_rows // samp_fact, new_cols // samp_fact, 8))

        # Apply block downsampling
        for i, ch in enumerate(chan):
            image_new[:, :, i] = block_reduce(image[:new_rows, :new_cols, ch], (samp_fact, samp_fact), np.mean)

        for i in range(8):
            reduction_func = np.median if i == 7 else np.mean
            qual_new[:, :, i] = block_reduce(qual[:new_rows, :new_cols, i], (samp_fact, samp_fact), reduction_func)

        return image_new, qual_new

    return image_sel, qual  # Return original if no downsampling


#function to make images out of the csv data frame
#epoch can be selected as number of sequence with time_axis_direct=True and time_el
#and as cloest to date with time_axis_direct=True adn time_date
#points can be plotted at specified locations with examples
def display_image_back(df,time_axis_direct=True,time_el=-1,exclude=2,examples=[],time_date=2024):
    x=int(df.x.max()+1)
    y=int(df.y.max()+1)
    time_is=df.groupby(df['time'])['time'].mean()
    if time_axis_direct==False:
        time_el=np.argmin(np.abs(time_is.iloc[:]-time_date))   
    #rounding added below otherwise not allways found
    df_sel=df[(np.round(df.time,5)==np.round(time_is.iloc[time_el],5))].loc[:,['coastal_blue','blue','turquoise','green','yellow','red','red_edge','infrared']]
    
    array=np.reshape(np.array(df_sel),(x,y,8))
    rgbBands = {3: [2,1,0], 4: [2,1,0], 8: [5,3,1]}
    nBands=8
    image_rgb = stretchIm(array, rgbBands[nBands], exclude,100-exclude)
    plt.figure(figsize=(10,10))
    plt.imshow(image_rgb[:,:,0:3])
    if len(examples)>0:
        for i in range(len(examples)):
            plt.plot(examples[i][0],examples[i][1],'o',color='red',ms=2)
    if time_axis_direct==False:        
        plt.title(f"image closest to {np.round(time_date,3)}")   
    else:
        if time_el>-0.5:
            plt.title(f"image {np.int32(time_el)} in time sequence")     
        else:
            plt.title(f"image {np.int32(-time_el)} in time sequence from the end") 


def plot_downsampl2(data, x, y, lim=0.5, start_dat=0, deltmag=1, quant=0.01, vert=0, lines=[2020], bands='rgbn'):
    # Define color mappings based on bands selection
    band_options = {
        'rgbn': ['blue', 'lime', 'red', 'gray'],
        'rgbn_edge': ['blue', 'lime', 'red', 'magenta', 'gray'],
        'all': ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray']
    }
    
    colors = band_options.get(bands, ['blue', 'lime', 'red', 'gray'])  # Default to 'rgbn'
    
    # Subset the data once for performance efficiency
    subset = data[(data.x == x) & (data.y == y)]
    
    if subset.empty:
        print("No data found for the given x and y coordinates.")
        return
    
    quality = subset['clear']
    dates = subset['time']
    
    # Select bands dynamically based on the input
    band_mapping = {
        'rgbn': ['blue', 'green', 'red', 'infrared'],
        'rgbn_edge': ['blue', 'green', 'red', 'red_edge', 'infrared'],
        'all': ['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared']
    }
    
    selected_bands = band_mapping.get(bands, ['blue', 'green', 'red', 'infrared'])
    values = subset[selected_bands]
    
    # Apply quality filtering
    valid_indices = quality > lim
    dates_sel = dates[valid_indices].iloc[start_dat:]
    data_sel = values[valid_indices].iloc[start_dat:]
    
    if dates_sel.empty or data_sel.empty:
        print("No valid data points after applying quality filtering.")
        return

    for i, color in enumerate(colors[:len(selected_bands)]):  # Ensure color matches selected bands
        band_data = data_sel.iloc[:, i]
        lims = np.quantile(band_data, [quant, 1 - quant])
        normalized_data = (band_data - lims[0]) / (lims[1] - lims[0]) + i * deltmag
        plt.plot(dates_sel, normalized_data, 'o', ms=2, color=color)

    # Plot vertical lines if requested
    pos = np.array([-0.02, 1 + (len(selected_bands) - 1) * deltmag + 0.02])    
    if vert > 0:
        for i in range(min(vert, len(lines))):
            plt.plot([lines[i], lines[i]], pos, 'k--')  # Dashed black line for vertical markers

    plt.ylim(pos[0], pos[1])
    plt.xlim(dates_sel.iloc[0] - 0.02, dates_sel.iloc[-1] + 0.02)  
    plt.xlabel("Time")
    plt.ylabel("Normalized Band Values")
    plt.title(f"Spectral Data for Location ({x}, {y})")
    plt.show()
    
#function for all the steps to transform from the 4d array to a 2d data frame

def transform_4d_2d(images, meta, dates, 
                        columns=['coastal_blue','blue','turquoise','green','yellow','red','red_edge','infrared',
                                'clear', 'snow','shadow','haze_light','haze_heavy', 'cloud','confidence','udm1',
                                'x','y','year','month','day','hour','minute','seconds','time']):
    """
    Transforms a 4D satellite imagery dataset into a 2D DataFrame.
    
    Parameters:
        images (numpy.ndarray): 4D array of satellite imagery with shape (H, W, Bands, Time)
        meta (numpy.ndarray): 4D array of metadata with shape (H, W, Metadata_Bands, Time)
        dates (numpy.ndarray): 3D array containing time-related information (H, W, 7, Time)
        columns (list): List of column names for the resulting DataFrame.

    Returns:
        pd.DataFrame: A 2D DataFrame where each row represents a pixel-time observation.
    """
    H, W, B, T = images.shape  # Get shape of the image data
    MB = meta.shape[2]          # Get the number of metadata bands

    # Generate X and Y coordinates using meshgrid
    x_coords, y_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Stack coordinate data
    coors = np.stack([x_coords, y_coords], axis=2)  # Shape: (H, W, 2)
    coors = np.repeat(coors[:, :, :, np.newaxis], T, axis=3)  # Expand to match time axis (H, W, 2, T)

    # Stack images, meta, and time info along the band axis
    all_data = np.concatenate((images, meta, coors, dates), axis=2)  # Shape: (H, W, B+MB+9, T)

    # Free up memory
    del images, meta
    gc.collect()

    # Reorder axes for reshaping
    all_data = np.moveaxis(all_data, 2, 3)  # Move bands to last axis (H, W, T, B+MB+9)

    # Reshape to 2D: (Total_Pixels * Time, Features)
    all_data = all_data.reshape(-1, all_data.shape[-1], order='A')

    # Convert to DataFrame
    df_all = pd.DataFrame.from_records(all_data, columns=columns)

    # Print summary statistics
    print(df_all.describe())

    return df_all, coors


def display_imageratio_back(df, exclude=2, examples=[], time_dates=[2020, 2021], band='red', limit=0.5, maxval=2):
    """
    Computes and displays the ratio of a spectral band at two selected times from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing image pixel data with 'x', 'y', 'time', and spectral bands.
        exclude (int): Unused parameter for now, kept for flexibility.
        examples (list): List of (x, y) points to highlight on the plot.
        time_dates (list): Two time points to compare.
        band (str): The spectral band to compute the ratio for.
        limit (float): Threshold for excluding flagged data (future implementation).
        maxval (float): Maximum cap for displayed ratio values.

    Returns:
        None (Displays the ratio image).
    """
    unique_times = np.sort(df['time'].unique())  # Get sorted unique times
    time_idx = [np.argmin(np.abs(unique_times - t)) for t in time_dates]  # Find closest available times

    # Select data for the closest time points
    df_sel1 = df[df['time'] == unique_times[time_idx[0]]].pivot(index='y', columns='x', values=band)
    df_sel2 = df[df['time'] == unique_times[time_idx[1]]].pivot(index='y', columns='x', values=band)

    # Convert to numpy arrays for visualization
    array1 = df_sel1.to_numpy()
    array2 = df_sel2.to_numpy()

    # Handle divide by zero and apply maxval limit
    ratio_array = np.where(array1 == 0, np.nan, array2 / array1)
    ratio_array = np.clip(ratio_array, 0, maxval)  # Avoid extreme outliers

    # Plot the ratio image
    plt.figure(figsize=(10, 10))
    plt.imshow(ratio_array.T, cmap='RdYlBu', interpolation='nearest')
    plt.colorbar(shrink=0.6, label=f"Ratio of {band} ({time_dates[1]}/{time_dates[0]})")

    # Plot example points
    if examples:
        examples = np.array(examples)  # Convert to array for easy indexing
        plt.scatter(examples[:, 0], examples[:, 1], color='red', s=20, marker='o', label='Example Points')

    plt.title(f"Ratio of {band} between {time_dates[1]} and {time_dates[0]}")
    plt.legend()
    plt.show()
    


def coords_el(coors2, tensor):
    """
    Finds the closest index in `tensor` for each coordinate in `coors`.

    Parameters:
    - coors2 (array-like): List of 2D coordinates (shape: Nx2). needs to be swaped 
    - tensor (numpy array): 2D array with shape (2, M) containing point coordinates.

    Returns:
    - list of indices corresponding to the nearest neighbors in `tensor`.
    """
    
    coors = np.zeros((len(coors2), 2))  # Ensure shape is (N, 2)
    for i in range(len(coors2)):
        coors[i,0]=coors2[i][1]
        coors[i,1]=coors2[i][0]        
    tensor_points = np.vstack((tensor[0, :], tensor[1, :])).T  # Convert tensor to (M, 2)

    # Use cKDTree for fast nearest neighbor search
    tree = cKDTree(tensor_points)
    _, nearest_indices = tree.query(coors)  # Query nearest points
    
    return nearest_indices.tolist()


def vis_outlier_regular(anomalies, xy, band_start=0, band_end=0, list_ex=[], list_type=[], add_title='',
                        direct=False, cmap='gist_earth_r', save=False, plot_name='test.png'):
    """
    Visualizes anomalies in a reshaped 2D grid.

    Parameters:
    - anomalies (numpy array): 3D anomaly data (Bands, X, Y) if direct=False; otherwise, a 2D array.
    - xy (numpy array): Coordinates of the grid.
    - band_start (int): Start band for summing anomalies.
    - band_end (int): End band for summing anomalies.
    - list_ex (list): List of indices to highlight.
    - list_type (list): List of types corresponding to `list_ex`.
    - add_title (str): Additional title for the plot.
    - direct (bool): If True, `anomalies` is used directly.
    - cmap (str): Colormap for visualization.
    - save (bool): Whether to save the plot as an image.
    - plot_name (str): Filename if `save` is True.

    Returns:
    - None (Displays the anomaly visualization).
    """
    if not direct:
        res6 = np.sum(anomalies[band_start:band_end + 1], axis=0)
    else:
        res6 = anomalies

    # Ensure `xy` provides the max extent
    grid_x, grid_y = int(np.max(xy[0]) + 1), int(np.max(xy[1]) + 1)
    comp1_b0 = np.reshape(res6, (grid_x, grid_y))

    # Plot anomaly heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(comp1_b0, cmap=cmap)
    plt.colorbar(shrink=0.6)

    # Determine plot title
    if not direct:
        title = f"Channel {band_start + 1}" if band_start == band_end else f"Channels {band_start + 1} to {band_end + 1}"
    else:
        title = add_title
    plt.title(title)

    # Collect and print outlier information
    outlier_info = []
    for i, ex in enumerate(list_ex):
        outlier_info.append(f"{list_type[i]} element {ex} has outlier classification {int(res6[ex])}")
    if outlier_info:
        print("\n".join(outlier_info))

    # Save plot if requested
    if save:
        plt.savefig(plot_name, bbox_inches='tight')
    
    plt.show()
        


def crop_geotiff(input_tiff, output_tiff, xmin, ymin, xmax, ymax):
    """
    Crops a GeoTIFF to the given bounding box (xmin, ymin, xmax, ymax).

    Parameters:
        input_tiff (str): Path to input GeoTIFF file.
        output_tiff (str): Path to save the cropped GeoTIFF.
        xmin (float): Minimum longitude (or X coordinate).
        ymin (float): Minimum latitude (or Y coordinate).
        xmax (float): Maximum longitude (or X coordinate).
        ymax (float): Maximum latitude (or Y coordinate).

    Returns:
        None (Saves cropped image to output_tiff).
    """
    with rasterio.open(input_tiff) as src:
        # Convert geographic coordinates to pixel indices
        col_start, row_start = map(int, ~src.transform * (xmin, ymax))
        col_stop, row_stop = map(int, ~src.transform * (xmax, ymin))

        # Ensure indices are within valid range
        col_start, col_stop = max(0, col_start), min(src.width, col_stop)
        row_start, row_stop = max(0, row_start), min(src.height, row_stop)

        # Define the window
        window = Window(col_start, row_start, width=col_stop - col_start, height=row_stop - row_start)

        # Read data within the window
        data = src.read(window=window)

        # Update transform for cropped area
        new_transform = from_origin(
            src.transform[2] + col_start * src.transform[0],  # New top-left X
            src.transform[5] + row_start * src.transform[4],  # New top-left Y
            src.transform[0],  # Pixel width
            src.transform[4]   # Pixel height
        )

        # Write cropped data to new file
        with rasterio.open(
            output_tiff, 'w', driver='GTiff',
            height=row_stop - row_start, width=col_stop - col_start,
            count=src.count, dtype=src.dtypes[0],
            crs=src.crs, transform=new_transform
        ) as dest:
            dest.write(data)

    print(f"Cropping complete. Output saved to {output_tiff}")    
    

def crop_image_coord(input_tiff, output_tiff, west_long, south_lat, east_long, north_lat):
    """
    Converts geographic coordinates (longitude, latitude) to UTM and crops the image.

    Parameters:
        input_tiff (str): Path to the input GeoTIFF file.
        output_tiff (str): Path to save the cropped GeoTIFF file.
        west_long (float): Western boundary in longitude.
        south_lat (float): Southern boundary in latitude.
        east_long (float): Eastern boundary in longitude.
        north_lat (float): Northern boundary in latitude.

    Returns:
        None (Saves the cropped image).
    """
    # Open the GeoTIFF file to retrieve CRS
    with rasterio.open(input_tiff) as src:
        print("CRS:", src.crs)
        print("Bounds:", src.bounds)
        print("Shape:", src.shape)
        print("Transform:", src.transform)
        print("No. of bands:", src.count)

        # Define transformation from WGS84 (Lat/Lon) to the raster's CRS
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        # Convert longitude/latitude to UTM
        westing, southing = transformer.transform(west_long, south_lat)
        easting, northing = transformer.transform(east_long, north_lat)

        print(f"Converted coordinate range: {easting, northing, westing, southing}")

    # Crop the GeoTIFF with transformed UTM coordinates
    crop_geotiff(input_tiff, output_tiff, westing, southing, easting, northing)
    


def plot_downsampl3(data, x, y, lim=0.5, start_dat=0, deltmag=1, quant=0.01, lines=[2020], bands='rgbn', exclude=2):
    """
    Plots time series spectral band values and overlays example pixels on RGB images.

    Parameters:
        data (pd.DataFrame): Multi-spectral data with time, bands, and quality flags.
        x (int): X coordinate of the point of interest.
        y (int): Y coordinate of the point of interest.
        lim (float): Quality threshold for selecting valid observations.
        start_dat (int): Start index for filtering time series.
        deltmag (float): Offset magnitude for band plotting.
        quant (float): Quantile threshold for normalizing band values.
        lines (list): List of time points for overlaying RGB images.
        bands (str): Spectral bands to visualize ('rgbn', 'all', 'rgbn_edge').
        exclude (int): Percentile exclusion for image stretching.

    Returns:
        None (Displays plots).
    """
    # Define band color mappings
    band_options = {
        'rgbn': ['blue', 'lime', 'red', 'gray'],
        'rgbn_edge': ['blue', 'lime', 'red', 'magenta', 'gray'],
        'all': ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray']
    }
    colors = band_options.get(bands, ['blue', 'lime', 'red', 'gray'])

    # Filter data **only once** for the given (x, y) location
    subset = data[(data.x == x) & (data.y == y)]

    if subset.empty:
        print(f"No data found for (x={x}, y={y}).")
        return

    quality = subset['clear']
    dates = subset['time']

    # Select spectral bands dynamically
    band_mapping = {
        'rgbn': ['blue', 'green', 'red', 'infrared'],
        'rgbn_edge': ['blue', 'green', 'red', 'red_edge', 'infrared'],
        'all': ['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared']
    }
    selected_bands = band_mapping.get(bands, ['blue', 'green', 'red', 'infrared'])
    values = subset[selected_bands]

    # Determine subplot count
    num_subplots = 2 if len(lines) <= 1 else 3
    fig, axs = plt.subplots(num_subplots, figsize=(10, 5 + (num_subplots - 1) * 3))

    # Time series plot
    for i, color in enumerate(colors[:len(selected_bands)]):
        dates_sel = dates[quality > lim].iloc[start_dat:]
        data_sel = values[quality > lim].iloc[start_dat:, i]

        if data_sel.empty:
            print(f"No valid data for {selected_bands[i]} at (x={x}, y={y}). Skipping plot.")
            continue

        lims = np.quantile(data_sel, [quant, 1 - quant])
        axs[0].plot(dates_sel, (data_sel - lims[0]) / (lims[1] - lims[0]) + i * deltmag, 'o', ms=2, color=color)

    axs[0].set_title(f"Time Series at (x={x}, y={y})")
    axs[0].set_ylabel("Normalized Band Values")
    
    # Add vertical time markers
    if lines:
        for line in lines:
            axs[0].axvline(line, linestyle="--", color="black")

    axs[0].set_xlim(dates_sel.iloc[0], dates_sel.iloc[-1])

    # Get max spatial extent
    xm, ym = int(data.x.max() + 1), int(data.y.max() + 1)
    examples = [y, x]  # Reverse order for imshow indexing

    # Plot RGB Images for Specified Years
    for idx, time_date in enumerate(lines[:num_subplots - 1]):
        # Find closest available time
        time_vals = np.sort(data['time'].unique())
        closest_time = time_vals[np.argmin(np.abs(time_vals - time_date))]

        # Filter data for selected time
        df_sel = data[np.isclose(data['time'], closest_time)]
        df_sel = df_sel[['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared']]

        if df_sel.empty:
            print(f"No data for selected time {closest_time}. Skipping RGB plot.")
            continue

        # Convert to numpy and reshape
        array = df_sel.to_numpy().reshape(xm, ym, 8)

        # Stretch image for visualization
        rgbBands = {3: [2, 1, 0], 4: [2, 1, 0], 8: [5, 3, 1]}
        nBands = 8
        if 'stretchIm' in globals():
            image_rgb = stretchIm(array, rgbBands[nBands], exclude, 100 - exclude)
        else:
            image_rgb = array[:, :, [5, 3, 1]]  # Default to unnormalized RGB (Red, Green, Blue)

        axs[idx + 1].imshow(image_rgb[:, :, :3])
        axs[idx + 1].scatter(examples[0], examples[1], color='red', marker='+', s=50)
        axs[idx + 1].set_title(f"RGB Image - {closest_time}")

    plt.tight_layout()
    plt.show()

def select_not_random(distances, anomalies, frac_tot_random2=0.2, frac_distance2=0.3, frac_ano2=0.5):
    """
    Selects an index based on weighted probability (random, max distance, or anomalies).
    
    Parameters:
        distances (numpy.ndarray): Array of max distances.
        anomalies (numpy.ndarray): Array indicating anomaly status.
        frac_tot_random2 (float): Fraction for purely random selection.
        frac_distance2 (float): Fraction for selecting max distance.
        frac_ano2 (float): Fraction for selecting an anomaly.

    Returns:
        tuple: (selected index, updated distances, updated anomalies)
    """
    # Normalize fractions to sum to 1
    total = frac_tot_random2 + frac_distance2 + frac_ano2
    frac_tot_random, frac_distance, frac_ano = frac_tot_random2 / total, frac_distance2 / total, frac_ano2 / total

    distances2 = np.copy(distances)
    anomalies2 = np.copy(anomalies)
    random1 = np.random.random()

    if random1 < frac_tot_random:
        # Random selection from non-anomalous elements (0 or 1)
        valid_indices = np.where((anomalies == 0) | (anomalies == 1))[0]
    elif random1 < frac_tot_random + frac_distance:
        # Select element with largest distance
        valid_indices = np.array([np.argmax(distances)])
    else:
        # Random selection from predicted anomalies (1)
        valid_indices = np.where(anomalies == 1)[0]

    if len(valid_indices) == 0:
        print("No valid selection found.")
        return None, distances2, anomalies2

    # Choose an index randomly from valid indices
    x = np.random.choice(valid_indices)

    # Mark as selected
    distances2[x] = 0
    anomalies2[x] = 0

    return x, distances2, anomalies2



def get_anomalies_predictions(data, anomalies, mode='slow'):
    """
    Computes Euclidean distances between all pixels and labeled anomalies,
    predicting new anomalies based on the nearest labeled neighbors.

    Parameters:
        data (pd.DataFrame): DataFrame containing time series for each pixel.
        anomalies (pd.DataFrame): Anomalies DataFrame with labeled anomalies.
        mode (str): Either 'slow' (iterative) or future optimized mode.

    Returns:
        dist_min (np.array): Minimum Euclidean distances.
        anoms_closest (np.array): Closest labeled anomaly for each pixel.
    """
    # Get unique sorted time values
    time_is = np.sort(data['time'].unique())

    # Get spatial grid dimensions
    xm, ym = int(data['x'].max() + 1), int(data['y'].max() + 1)

    # Initialize arrays
    res = np.zeros((xm * ym, len(time_is) * 8))  # All pixel data
    res_an = np.zeros((anomalies[anomalies['anom_truth'] > -1].shape[0], len(time_is) * 8))  # Labeled anomalies

    # Fill `res` and `res_an`
    if mode == 'slow':
        for i, t in enumerate(time_is):
            # Get band values for the given time
            df_sel = data[data['time'] == t][['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared']].to_numpy()
            res[:, i * 8: (i + 1) * 8] = df_sel

            # Extract corresponding labeled anomalies
            an_sel = df_sel[anomalies['anom_truth'] > -1, :]
            res_an[:, i * 8: (i + 1) * 8] = an_sel

    # Normalize data (handle zero standard deviation)
    std = np.std(res, axis=0)
    mean = np.mean(res, axis=0)
    std = np.where(std == 0, 1, std)  # Prevent division by zero
    res = (res - mean) / std
    res_an = (res_an - mean) / std

    # Compute Euclidean distances (efficiently)
    dist = cdist(res, res_an, metric='euclidean')

    # Find nearest anomaly for each pixel
    dist_amin = np.argmin(dist, axis=1)
    dist_min = np.min(dist, axis=1)

    # Extract labeled anomalies
    labelled = anomalies[anomalies['anom_truth'] > -1]['anom_truth'].to_numpy()

    # Assign closest anomaly class
    anoms_closest = np.where(anomalies['anom_truth'] > -1, 2 * (anomalies['anom_truth'] - 0.25), labelled[dist_amin])

    return dist_min, anoms_closest

def get_xy(dat, element):
    """
    Converts a 1D index into (x, y) coordinates in a 2D grid.

    Parameters:
        dat (pd.DataFrame): DataFrame containing 'x' and 'y' columns.
        element (int): Linear index to be converted.

    Returns:
        tuple: (x, y) coordinates in the grid.
    """
    # Get grid dimensions dynamically
    xm = dat['x'].nunique()  # Number of unique x values
    ym = dat['y'].nunique()  # Number of unique y values

    # Validate input range
    if element < 0 or element >= xm * ym:
        raise ValueError(f"Element index {element} is out of bounds for grid size ({xm}, {ym})")

    # Compute (x, y) using divmod for efficiency
    x, y = divmod(element, ym)

    return x, y


#works currently only for down sample data, later  needs to be changed 

def plot_time(data, element, lim=0.5, start_dat=0, deltmag=1, quant=0.01, vert=0, lines=[2020], bands='rgbn'):
    """
    Plots the time series of spectral band values for a selected pixel.

    Parameters:
        data (pd.DataFrame): DataFrame containing spectral data.
        element (int): Index of the pixel in a flattened 2D grid.
        lim (float): Quality threshold for selecting valid data.
        start_dat (int): Starting index for filtering.
        deltmag (float): Offset magnitude for plotting multiple bands.
        quant (float): Quantile threshold for normalizing band values.
        vert (int): Number of vertical reference lines.
        lines (list): Time values where vertical lines should be drawn.
        bands (str): Spectral bands to plot ('rgbn', 'all', 'rgbn_edge').

    Returns:
        None (Displays the plot).
    """
    # Define band color mappings
    band_options = {
        'rgbn': (['blue', 'green', 'red', 'infrared'], ['blue', 'lime', 'red', 'gray']),
        'rgbn_edge': (['blue', 'green', 'red', 'red_edge', 'infrared'], ['blue', 'lime', 'red', 'magenta', 'gray']),
        'all': (['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared'], 
                ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray'])
    }
    selected_bands, colors = band_options.get(bands, (['blue', 'green', 'red', 'infrared'], ['blue', 'lime', 'red', 'gray']))

    # Get (x, y) coordinates
    x, y = get_xy(data, element)

    # Filter data **once**
    subset = data[(data['x'] == x) & (data['y'] == y)]

    if subset.empty:
        print(f"No data found for (x={x}, y={y}).")
        return

    quality = subset['clear']
    dates = subset['time']
    values = subset[selected_bands]

    # Select valid dates based on quality threshold
    valid_indices = quality > lim
    dates_sel = dates[valid_indices].iloc[start_dat:]
    values_sel = values[valid_indices].iloc[start_dat:]

    if dates_sel.empty:
        print(f"No valid data for (x={x}, y={y}) after quality filtering.")
        return

    # Plot spectral time series
    plt.figure(figsize=(10, 5))
    
    for i, color in enumerate(colors[:len(selected_bands)]):
        data_sel = values_sel.iloc[:, i]
        lims = np.quantile(data_sel, [quant, 1 - quant])
        normalized_data = (data_sel - lims[0]) / (lims[1] - lims[0]) + i * deltmag
        plt.plot(dates_sel, normalized_data, 'o', ms=2, color=color, label=selected_bands[i])

    # Add vertical reference lines
    for line in lines[:vert]:
        plt.axvline(x=line, linestyle="--", color="black")

    # Set plot limits
    plt.ylim(-0.02, 1 + (len(colors) - 1) * deltmag + 0.02)
    plt.xlim(dates_sel.iloc[0], dates_sel.iloc[-1])

    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("Normalized Band Values")
    plt.title(f"Spectral Time Series at (x={x}, y={y})")
    plt.legend()
    plt.show()
    

def plot_time_space(data, element, lim=0.5, start_dat=0, deltmag=1, quant=0.01, lines=[2020], bands='rgbn', exclude=2):
    """
    Plots time-series spectral band values and spatial images for a selected pixel.

    Parameters:
        data (pd.DataFrame): DataFrame containing spectral data.
        element (int): Index of the pixel in a flattened 2D grid.
        lim (float): Quality threshold for selecting valid data.
        start_dat (int): Starting index for filtering.
        deltmag (float): Offset magnitude for plotting multiple bands.
        quant (float): Quantile threshold for normalizing band values.
        lines (list): Time values where images should be plotted.
        bands (str): Spectral bands to plot ('rgbn', 'all', 'rgbn_edge').
        exclude (int): Percentile exclusion for image stretching.

    Returns:
        None (Displays the plot).
    """
    # Define band color mappings
    band_options = {
        'rgbn': (['blue', 'green', 'red', 'infrared'], ['blue', 'lime', 'red', 'gray']),
        'rgbn_edge': (['blue', 'green', 'red', 'red_edge', 'infrared'], ['blue', 'lime', 'red', 'magenta', 'gray']),
        'all': (['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared'], 
                ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray'])
    }
    selected_bands, colors = band_options.get(bands, (['blue', 'green', 'red', 'infrared'], ['blue', 'lime', 'red', 'gray']))

    # Get (x, y) coordinates
    x, y = get_xy(data, element)

    # Filter data **once**
    subset = data[(data['x'] == x) & (data['y'] == y)]
    
    if subset.empty:
        print(f"No data found for (x={x}, y={y}).")
        return

    quality = subset['clear']
    dates = subset['time']
    values = subset[selected_bands]

    # Select valid dates based on quality threshold
    valid_indices = quality > lim
    dates_sel = dates[valid_indices].iloc[start_dat:]
    values_sel = values[valid_indices].iloc[start_dat:]

    if dates_sel.empty:
        print(f"No valid data for (x={x}, y={y}) after quality filtering.")
        return

    # Determine subplot count
    num_subplots = 2 if len(lines) <= 1 else 3
    fig, axs = plt.subplots(num_subplots, figsize=(10, 5 + (num_subplots - 1) * 3))

    # Plot spectral time series
    for i, color in enumerate(colors[:len(selected_bands)]):
        data_sel = values_sel.iloc[:, i]
        lims = np.quantile(data_sel, [quant, 1 - quant])
        normalized_data = (data_sel - lims[0]) / (lims[1] - lims[0]) + i * deltmag
        axs[0].plot(dates_sel, normalized_data, 'o', ms=2, color=color, label=selected_bands[i])

    axs[0].set_title(f"Spectral Time Series at (x={x}, y={y})")
    axs[0].set_ylabel("Normalized Band Values")
    axs[0].legend()

    # Add vertical reference lines
    for line in lines:
        axs[0].axvline(x=line, linestyle="--", color="black")

    # Set plot limits
    axs[0].set_xlim(dates_sel.iloc[0], dates_sel.iloc[-1])
    axs[0].set_ylim(-0.02, 1 + (len(colors) - 1) * deltmag + 0.02)

    # Get spatial dimensions
    xm, ym = int(data['x'].max() + 1), int(data['y'].max() + 1)
    examples = [y, x]  # Reverse order for imshow indexing

    # Get sorted time values
    time_vals = np.sort(data['time'].unique())

    # Plot RGB Images for Selected Years
    for idx, time_date in enumerate(lines[:num_subplots - 1]):
        # Find closest available time
        closest_time = time_vals[np.argmin(np.abs(time_vals - time_date))]
        
        df_sel = data[(np.round(data.time,5) == np.round(closest_time,5))].loc[:, ['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared']]
        if df_sel.empty:
            print(f"No data for selected time {closest_time}. Skipping RGB plot.")
            continue
            
        df_sel = np.reshape(np.array(df_sel), (xm, ym, 8))
        # reframe to numpy, not really needed now, but kept 
        try:
            array = df_sel
        except ValueError:
            print(f"Reshaping failed: Expected ({xm}, {ym}, 8), but got {df_sel.shape}")
            return

        # Stretch image for visualization
        rgbBands = {3: [2, 1, 0], 4: [2, 1, 0], 8: [5, 3, 1]}
        nBands = 8
        if 'stretchIm' in globals():
            image_rgb = stretchIm(array, rgbBands[nBands], exclude, 100 - exclude)
        else:
            image_rgb = array[:, :, [5, 3, 1]]  # Default to unnormalized RGB (Red, Green, Blue)

        axs[idx + 1].imshow(image_rgb[:, :, :3])
        axs[idx + 1].scatter(examples[0], examples[1], color='red', marker='+', s=50)
        axs[idx + 1].set_title(f"RGB Image - {np.round(closest_time,4)}")

    plt.tight_layout()
    plt.show()

def make_anomalies_plot(file_name):
    a=pd.read_csv(file_name)
    a['anomalosity']=np.where(a.status=='Anomalous', 1,np.where(a.status=='Not_anomalous',0,0.5))
    an2=file_name.split('/')
    an=an2[-1].split('_')
    target=None
    if an[1]=='armazones':
        target='armazones'
    elif an[1]=='forest':
        target='forest_hills'      
    elif an[1]=='foresthill':
        target='forest_hills'          
    elif an[1]=='northepoint':
        target='northe_point' 
    elif an[1]=='northe':
        target='northe_point'   
    elif an[1]=='np':
        target='northe_point'         
    elif an[1]=='lapalma':
        target='la_palma'     
    elif an[1]=='pokrovsk':
        target='pokrovsk'   
    elif an[1]=='councilbluffs':
        target='council_bluffs'    
    elif an[1]=='kakamega':
        target='kakamega'   
    elif an[1]=='limpopo':
        target='limpopo' 
        
    if target=='armazones':
        #now just  inserted not file loaded to get image size
        xm= int(64 + 1)
        ym= int(81 + 1)
        dat=0        
    elif target=='forest_hills':    
        xm= int(73 + 1)
        ym= int(133 + 1)

    elif target=='northe_point':    
        xm= int(67+ 1)
        ym= int(117 + 1)  
    elif target=='la_palma':    
        xm= int(191+ 1)
        ym= int(510 + 1)       
    elif target=='council_bluffs':    
        xm= int(77+ 1)
        ym= int(137 + 1)  
    elif target=='surf':    
        xm= int(114+ 1)
        ym= int(171 + 1)  
    elif target=='limpopo':    
        xm= int(82+ 1)
        ym= int(128 + 1)  
    elif target=='kakamega':    
        xm= int(103+ 1)
        ym= int(198 + 1)  
    elif target=='pokrovsk':    
        xm= int(152+ 1)
        ym= int(131 + 1)  
        
    anoms_in_ar=np.ones((xm,ym))*0.5
    for i in range(a.shape[0]):
        anoms_in_ar[int(a.x.iloc[i]),int(a.y.iloc[i])]=a.anomalosity.iloc[i]
    plt.title(f"{an[0]} {target}")
    plt.imshow(anoms_in_ar)
    plt.colorbar(label="anomalosity", orientation="horizontal",aspect=60,pad=0.14)    
    
    
def make_anomalies_plot_comb(list_names,target_is='armazones',measure='mean',cmap='cool'):
    if target_is=='armazones':
        #now just  inserted not file loaded to get image size
        xm= int(64 + 1)
        ym= int(81 + 1)
        dat=0        
    elif target_is=='forest_hills':    
        xm= int(73 + 1)
        ym= int(133 + 1)

    elif target_is=='northe_point':    
        xm= int(67+ 1)
        ym= int(117 + 1)  
    elif target_is=='la_palma':    
        xm= int(191+ 1)
        ym= int(510 + 1)       
    elif target_is=='council_bluffs':    
        xm= int(77+ 1)
        ym= int(137 + 1)  
    elif target_is=='surf':    
        xm= int(114+ 1)
        ym= int(171 + 1)  
    elif target_is=='limpopo':    
        xm= int(82+ 1)
        ym= int(128 + 1)  
    elif target_is=='kakamega':    
        xm= int(103+ 1)
        ym= int(198 + 1)  
    elif target_is=='pokrovsk':    
        xm= int(152+ 1)
        ym= int(131 + 1)  

    counter=0
    list_df=[]
    for j in range(len(list_names)):
        a=pd.read_csv(list_names[j])
        a['anomalosity']=np.where(a.status=='Anomalous', 1,np.where(a.status=='Not_anomalous',0,0.5))
        an2=list_names[j].split('/')
        an=an2[-1].split('_')
        target=None
        if an[1]=='armazones':
            target='armazones'
        elif an[1]=='forest':
            target='forest_hills'      
        elif an[1]=='foresthill':
            target='forest_hills'          
        elif an[1]=='northepoint':
            target='northe_point' 
        elif an[1]=='northe':
            target='northe_point'   
        elif an[1]=='np':
            target='northe_point'              
        elif an[1]=='lapalma':
            target='la_palma'
        elif an[1]=='pokrovsk':
            target='pokrovsk'   
        elif an[1]=='councilbluffs':
            target='council_bluffs'        
        elif an[1]=='limpopo':
            target='limpopo'   
        elif an[1]=='kakamega':
            target='kakamega'               
        if target==target_is:
            list_df.append(a)
            counter+=1
    anoms_in_ar=np.ones((xm,ym,counter))*0.5 
    anoms_in_ar[:,:,:]=np.nan
    anoms_in_art=np.ones((xm,ym,counter))*0.5 
    anoms_in_art[:,:,:]=np.nan    
    anoms_in_art2=np.ones((xm,ym,counter))*0.5 
    anoms_in_art2[:,:,:]=np.nan      
    counter=0    
    for j in range(len(list_df)):
        for i in range(list_df[j].shape[0]):
            anoms_in_ar[int(list_df[j].x.iloc[i]),int(list_df[j].y.iloc[i]),int(counter)]=list_df[j].anomalosity.iloc[i]
            if list_df[j].anomalosity.iloc[i]>0.5:
                anoms_in_art[int(list_df[j].x.iloc[i]),int(list_df[j].y.iloc[i]),int(counter)]=list_df[j].vert_line_x.iloc[i]
                anoms_in_art2[int(list_df[j].x.iloc[i]),int(list_df[j].y.iloc[i]),int(counter)]=list_df[j].second_vert_line_x.iloc[i]
        counter+=1    
    if measure=='mean' or measure=='median' or  measure=='max' or measure=='count':
        measure2=measure
    elif measure=='mean_time':
        measure2='mean'       
    elif measure=='median_time':
        measure2='median' 
    elif measure=='max_time':
        measure2='max'     
    elif measure=='sec_mean_time':
        measure2='mean'       
    elif measure=='sec_median_time':
        measure2='median' 
    elif measure=='sec_max_time':
        measure2='max'       
    elif measure=='count_time':
        measure2='count'     
    elif measure=='sec_count_time':
        measure2='count'        
    plt.title(f"{target_is } {len(list_df)} lists {measure2}")
    if measure=='mean':
        plt.imshow(np.nanmean(anoms_in_ar,axis=2),cmap=cmap)
    elif measure=='mean_time':
        plt.imshow(np.nanmean(anoms_in_art,axis=2),cmap=cmap)    
    elif measure=='max_time':
        plt.imshow(np.nanmax(anoms_in_art,axis=2),cmap=cmap)
    elif measure=='median_time':
        plt.imshow(np.nanmedian(anoms_in_art,axis=2),cmap=cmap)     
    elif measure=='sec_mean_time':
        plt.imshow(np.nanmean(anoms_in_art2,axis=2),cmap=cmap)    
    elif measure=='sec_max_time':
        plt.imshow(np.nanmax(anoms_in_art2,axis=2),cmap=cmap)
    elif measure=='sec_median_time':
        plt.imshow(np.nanmedian(anoms_in_art2,axis=2),cmap=cmap)          
    elif measure=='max':
        plt.imshow(np.nanmax(anoms_in_ar,axis=2),cmap=cmap)        
    elif measure=='median':
        plt.imshow(np.nanmedian(anoms_in_ar,axis=2),cmap=cmap)   
    elif measure=='count':
        plt.imshow((~np.isnan(anoms_in_ar)).sum(axis=2),cmap=cmap)  
    elif measure=='count_time':
        plt.imshow((~np.isnan(anoms_in_art)).sum(axis=2),cmap=cmap)  
    elif measure=='sec_count_time':
        plt.imshow((~np.isnan(anoms_in_art2)).sum(axis=2),cmap=cmap)          
    if measure=='mean' or  measure=='median' or measure=='max':    
        plt.colorbar(label="anomalosity", orientation="horizontal",aspect=60,pad=0.14)  
    elif measure=='mean_time' or  measure=='median_time' or measure=='max_time':    
        plt.colorbar(label="time of first anomaly", orientation="horizontal",aspect=60,pad=0.14)      
    elif measure=='sec_mean_time' or  measure=='sec_median_time' or measure=='sec_max_time':    
        plt.colorbar(label="time of second anomaly", orientation="horizontal",aspect=60,pad=0.14)            
    elif measure=='count':
        plt.colorbar(label="number of values", orientation="horizontal",aspect=60,pad=0.14) 
    elif measure=='count_time':
        plt.colorbar(label="number of anomaly times 1", orientation="horizontal",aspect=60,pad=0.14) 
    elif measure=='sec_count_time':
        plt.colorbar(label="number of anomaly times 2", orientation="horizontal",aspect=60,pad=0.14)             

#transform data for euclidean distances 
def transform_to_2d(sel_fluxes,coor,with_time=False):
    if with_time==False:
        columns=['coastal_blue','blue','turquoise','green','yellow','red','red_edge','infrared']
    else:
        columns=['coastal_blue','blue','turquoise','green','yellow','red','red_edge','infrared','time']
    sel1=np.array(sel_fluxes[(np.round(sel_fluxes.x)==np.round(coor[0,0])) & (np.round(sel_fluxes.y)==np.round(coor[0,1]))].loc[:,columns])
    sel2=np.reshape(np.array(sel1),(sel1.shape[0]*sel1.shape[1]))
    main=np.zeros((sel2.shape[0],coor.shape[0]))
    for l in range(coor.shape[0]):
        sel1=np.array(sel_fluxes[(np.round(sel_fluxes.x)==np.round(coor[l,0])) & (np.round(sel_fluxes.y)==np.round(coor[l,1]))].loc[:,columns])
        main[:,l]=np.reshape(np.array(sel1),(sel1.shape[0]*sel1.shape[1]))
    return main

def xy_to_id(xy,target=None,ym=0):
    """
    converts coordinates of pixel [x,y] to index for the usual reshaping

    Parameters:
        xy (list if two int): target pixel
        target (selected strings): the known targets, gets ym for them
        ym (integer): y extension of image can be given as alternative, one of them is needed
    Returns:
         Index (Integer) is for a single time step not all together)
    """    
    if target=='armazones':
        ym= int(81 + 1)      
    elif target=='forest_hills':    
        ym= int(133 + 1)
    elif target=='northe_point':    
        ym= int(117 + 1)  
    elif target=='la_palma':    
        ym= int(510 + 1)       
    elif target=='council_bluffs':    
        ym= int(137 + 1)  
    elif target=='surf':    
        ym= int(171 + 1)  
    elif target=='limpopo':    
        ym= int(128 + 1)  
    elif target=='kakamega':    
        ym= int(198 + 1)  
    elif target=='pokrovsk':    
        ym= int(131 + 1)  
    pix=xy[0]+xy[1]*ym
    return pix



def plot_time_space_delt(data, element,delt_display=600,el_given=True,xy=[0,0], lim=0.5, deltpos=0,start_dat=0, deltmag=1, quant=0.01, lines=[2020], bands='rgbn', exclude=2):
    """
    Plots time-series spectral band values and spatial images for a selected pixel or pixel difference

    Parameters:
        data (pd.DataFrame): DataFrame containing spectral data.
        element (int): Index of the pixel in a flattened 2D grid.
        lim (float): Quality threshold for selecting valid data. Now >=lim that 0 is possible. 
        start_dat (int): Starting index for filtering.
        deltmag (float): Offset magnitude for plotting multiple bands.
        quant (float): Quantile threshold for normalizing band values.
        lines (list): Time values where images should be plotted.
        bands (str): Spectral bands to plot ('rgbn', 'all', 'rgbn_edge').
        exclude (int): Percentile exclusion for image stretching.
        deltposm (int greater equals 0); deltpos positions used for differentional, when 0 just flux
        el_given (Bool): whether element is used or coordinates
        xy (list of two integers): coordinates used for display
        deltpos (0 or positive int): if greater zero not the direct  lightcurve is shown but the difference between the target and pixels  deltpos distant
        deltdisplay (positive int): box radius of displayed image region, default results in full image 
    Returns:
        None (Displays the plot).
    """
    # Define band color mapping
    band_options = {
        'rgbn': (['blue', 'green', 'red', 'infrared'], ['blue', 'lime', 'red', 'gray']),
        'rgbn_edge': (['blue', 'green', 'red', 'red_edge', 'infrared'], ['blue', 'lime', 'red', 'magenta', 'gray']),
        'all': (['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared'], 
                ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray'])
    }
    selected_bands, colors = band_options.get(bands, (['blue', 'green', 'red', 'infrared'], ['blue', 'lime', 'red', 'gray']))

    # Get (x, y) coordinates
    if el_given==True:
        x, y = get_xy(data, element)
    else:
        x=xy[0]
        y=xy[1]
        element=xy_to_id(xy,ym=int(data.y.max()+1))
        
    #max maximal x, y 
    xm=data.x.max()
    ym=data.y.max()    
    # Filter data **once**
    subset = data[(data['x'] == x) & (data['y'] == y)]
    #get delta positions, 4 others in /y plus minus deltpos
    if deltpos>0:
        deltsubsets=[]
        npos=np.zeros((2,4))
        npos[0,0]=x+deltpos
        npos[0,1]=x-deltpos
        npos[1,0:2]=y
        npos[0,2:4]=x
        npos[1,2]=y+deltpos
        npos[1,3]=y-deltpos
        for i in range(npos.shape[1]):
            if npos[0,i]>=0 and npos[0,i]<=xm and  npos[1,i]>=0 and npos[1,i]<=ym: 
                deltsubsets.append(data[(data['x'] == npos[0,i]) & (data['y'] == npos[1,i])])    
    if subset.empty:
        print(f"No data found for (x={x}, y={y}).")
        return

    quality = subset['clear']
    dates = subset['time']
    values = subset[selected_bands]
    if deltpos>0:
        deltvalues=[]
        #get values for deltas no time or fag done main one used 
        for i in range(len(deltsubsets)):
            deltvalues.append(deltsubsets[i][selected_bands])
        
    # Select valid dates based on quality threshold
    valid_indices = quality >= lim
    dates_sel = dates[valid_indices].iloc[start_dat:]
    values_sel = values[valid_indices].iloc[start_dat:]
    if deltpos>0:
        vdeltvalues=[]
        #get values for deltas no time or fag done main one used 
        for i in range(len(deltvalues)):
            #using numoy array now because indexes not correct but order is the the same 
            vdeltvalues.append(np.array(deltvalues[i])[valid_indices,start_dat:])
        tot=np.array(vdeltvalues)
        #take average of the delta
        av_delt=np.mean(tot,axis=0)
    if dates_sel.empty:
        print(f"No valid data for (x={x}, y={y}) after quality filtering.")
        return

    # Determine subplot count
    num_subplots = 2 if len(lines) <= 1 else 3
    fig, axs = plt.subplots(num_subplots, figsize=(10, 5 + (num_subplots - 1) * 3))

    # Plot spectral time series
    for i, color in enumerate(colors[:len(selected_bands)]):
        if deltpos==0:
            data_sel = values_sel.iloc[:, i]
        else:
            data_sel =values_sel.iloc[:,i]-av_delt[:,i]
            
        lims = np.quantile(data_sel, [quant, 1 - quant])
        normalized_data = (data_sel - lims[0]) / (lims[1] - lims[0]) + i * deltmag
        axs[0].plot(dates_sel, normalized_data, 'o', ms=2, color=color, label=selected_bands[i])

    axs[0].set_title(f"Spectral Time Series at (x={x}, y={y})")
    axs[0].set_ylabel("Normalized Band Values")
    axs[0].legend()

    # Add vertical reference lines
    for line in lines:
        axs[0].axvline(x=line, linestyle="--", color="black")

    # Set plot limits
    axs[0].set_xlim(dates_sel.iloc[0], dates_sel.iloc[-1])
    axs[0].set_ylim(-0.02, 1 + (len(colors) - 1) * deltmag + 0.02)

    # Get spatial dimensions
    xm, ym = int(data['x'].max() + 1), int(data['y'].max() + 1)
    examples = [y, x]  # Reverse order for imshow indexing

    # Get sorted time values
    time_vals = np.sort(data['time'].unique())

    # Plot RGB Images for Selected Years
    for idx, time_date in enumerate(lines[:num_subplots - 1]):
        # Find closest available time
        closest_time = time_vals[np.argmin(np.abs(time_vals - time_date))]
        
        df_sel = data[(np.round(data.time,5) == np.round(closest_time,5))].loc[:, ['coastal_blue', 'blue', 'turquoise', 'green', 'yellow', 'red', 'red_edge', 'infrared']]
        if df_sel.empty:
            print(f"No data for selected time {closest_time}. Skipping RGB plot.")
            continue
            
        df_sel = np.reshape(np.array(df_sel), (xm, ym, 8))
        # reframe to numpy, not really needed now, but kept 
        try:
            array = df_sel
        except ValueError:
            print(f"Reshaping failed: Expected ({xm}, {ym}, 8), but got {df_sel.shape}")
            return

        # Stretch image for visualization
        rgbBands = {3: [2, 1, 0], 4: [2, 1, 0], 8: [5, 3, 1]}
        nBands = 8
        if 'stretchIm' in globals():
            image_rgb = stretchIm(array, rgbBands[nBands], exclude, 100 - exclude)
        else:
            image_rgb = array[:, :, [5, 3, 1]]  # Default to unnormalized RGB (Red, Green, Blue)
        minx=max(x-delt_display,0)
        miny=max(y-delt_display,0)
        maxx=min(x+delt_display,xm)
        maxy=min(y+delt_display,ym)
        axs[idx + 1].imshow(image_rgb[minx:maxx, miny:maxy, :3])
        axs[idx + 1].scatter(examples[0]-miny, examples[1]-minx, color='red', marker='+', s=50)
        axs[idx + 1].set_title(f"RGB Image - {np.round(closest_time,4)}")
        #how to zoom on object of interest 
    plt.tight_layout()
    plt.show()


def combine_labels(list_files,dat,clear_lim=0.5,norm=True,test_plot=False):
    persons=['devon','jonathan','kiana','tobias','zach']
    sel_clear=dat.groupby(dat.time)['clear'].mean()
    dic=sel_clear.to_dict()
    dat['clear_epoch']=dat['time'].map(dic)
    sel_fluxes=dat[(dat.clear_epoch>clear_lim)]
    dates,xy,other,meta=reshape_for_ccdc(sel_fluxes,mode='slow') 
    dates=0
    meta=0
    #labels
    labs_a=np.zeros((int(np.max(xy[0])+1),int(np.max(xy[1])+1),len(persons)))
    #date 1
    date_a=np.zeros((int(np.max(xy[0])+1),int(np.max(xy[1])+1),len(persons)))
    #date 2
    date_b=np.zeros((int(np.max(xy[0])+1),int(np.max(xy[1])+1),len(persons)))    
    #distances
    dist_a=np.zeros((int(np.max(xy[0])+1),int(np.max(xy[1])+1),len(persons)))
    # index of distance
    index_a=np.zeros((int(np.max(xy[0])+1),int(np.max(xy[1])+1),len(persons)))    
    for i in range(len(persons)):
        print(persons[i])
        list_dat=[]
        for j in range(len(list_files)):
            if persons[i] in  list_files[j]: 
                list_dat.append(pd.read_csv(list_files[j]))
        print(f"number of lists {len(list_dat)}")
        #only when there are labels of the person, not for zero
        if len(list_dat)>0:
            labelled=pd.concat(list_dat)
            labelled=labelled[(labelled.status=='Anomalous') | (labelled.status=='Not_anomalous')]
            labelled['anomalosity']=np.where(labelled.status=='Anomalous', 1,0)
            print(f"mean anomalosity={np.round(labelled.anomalosity.mean(),3)}")
            coor=np.array(labelled.loc[:,['x','y']])
            main2=transform_to_2d(sel_fluxes,coor,with_time=False)
            main=np.ones((main2.shape[0],main2.shape[1]))
            for j in range(main2.shape[0]):
                new=(j%other.shape[0])*other.shape[1]+int(j/other.shape[0])%other.shape[1]
                main[new,:]=main2[j,:]
            other2=np.reshape(other,(other.shape[0]*other.shape[1],other.shape[2]))
            if test_plot==True:
                plt.plot(range(main.shape[0]),main[:,0],'o',ms=2,label='labelled')
                plt.plot(range(other2.shape[0]),other2[:,0],'o',ms=2,label='all')
                plt.legend()
            #add height normalization maybe later assumes that absolute level does not matter much     
            if norm==True:
                std=np.std(other2,axis=1)
                mea=np.std(other2,axis=1)
                #omit normalization as test 
                other2=(other2-mea[:, np.newaxis])/std[:, np.newaxis]
                main=(main-mea[:, np.newaxis])/std[:, np.newaxis]
            dist1=euclidean_distances(other2.T, main.T)
            rmin=np.argmin(dist1,axis=1)
            rmin2=np.min(dist1,axis=1)
            labs_a[:,:,i]=np.reshape(np.array(labelled.anomalosity.iloc[rmin]), (int(np.max(xy[0])+1),int(np.max(xy[1])+1)))
            date_a[:,:,i]=np.reshape(np.array(labelled.vert_line_x.iloc[rmin]), (int(np.max(xy[0])+1),int(np.max(xy[1])+1)))
            date_b[:,:,i]=np.reshape(np.array(labelled.second_vert_line_x.iloc[rmin]), (int(np.max(xy[0])+1),int(np.max(xy[1])+1)))
            dist_a[:,:,i]=np.reshape(np.array(rmin2), (int(np.max(xy[0])+1),int(np.max(xy[1])+1)))
            index_a[:,:,i]=np.reshape(np.array(rmin), (int(np.max(xy[0])+1),int(np.max(xy[1])+1)))
    return labs_a,  xy, dist_a,date_a, date_b,index_a

#here concat all of one target and get distances of all pixels compared to it 
def get_most_extreme_index(list_in,add_name='_more_targets_v2.csv',fluxes=0,mode='predict1',clear_lim=0.5,norm=True,n_out=100,test_plot=False):
    list_dat=[]
    for i in range(len(list_in)):
        list_dat.append(pd.read_csv(list_in[i]))
        list_dat[-1]['anomalosity']=np.where(list_dat[-1].status=='Anomalous', 1,np.where(list_dat[-1].status=='Not_anomalous',0,0.5))
    df=pd.concat(list_dat)
    an=list_in[0].split('_')
    if an[1]=='armazones':
        target='armazones'
    elif an[1]=='forest':
        target='forest_hills'      
    elif an[1]=='foresthill':
        target='forest_hills'          
    elif an[1]=='northepoint':
        target='northe_point' 
    elif an[1]=='northe':
        target='northe_point' 
    elif an[1]=='surf':
        target='surf'    
    elif an[1]=='councilbluffs':
        target='councilbluffs'     
    elif an[1]=='kakamega':
        target='kakamega'    
    elif an[1]=='limpopo':
        target='limpopo'           
    print(target)   
    if mode=='predict1':
        #select clear 
        sel_clear=fluxes.groupby(fluxes.time)['clear'].mean()
        dic=sel_clear.to_dict()
        fluxes['clear_epoch']=fluxes['time'].map(dic)
        sel_fluxes=fluxes[(fluxes.clear_epoch>clear_lim)]
    
    list_all=[]
    #iterating over lists
    for i in range(len(list_in)):
        an2=list_in[i].split('_')
        print(an2[0])
        list_all.append(list_dat[i])
    fin=pd.concat(list_all)
    coor=np.array(fin.loc[:,['x','y']])
    main2=transform_to_2d(sel_fluxes,coor,with_time=False)

    dates,xy,other,meta=reshape_for_ccdc(sel_fluxes,mode='slow')

    dates=0
    meta=0
    main=np.ones((main2.shape[0],main2.shape[1]))
    for i in range(main2.shape[0]):
        new=(i%other.shape[0])*other.shape[1]+int(i/other.shape[0])%other.shape[1]
        main[new,:]=main2[i,:]
    other=np.reshape(other,(other.shape[0]*other.shape[1],other.shape[2]))
    if test_plot==True:
        plt.plot(range(main.shape[0]),main[:,0],'o',ms=2)
        plt.plot(range(other.shape[0]),other[:,0],'o',ms=2)    
    #is now same order in both, now getting distances
    if norm==True:
        std=np.std(other,axis=1)
        mea=np.std(other,axis=1)
        #omit normalization as test 
        other=(other-mea[:, np.newaxis])/std[:, np.newaxis]
        main=(main-mea[:, np.newaxis])/std[:, np.newaxis]
    dist1=euclidean_distances(other.T, main.T)
    rmin=np.min(dist1,axis=1)
    s=np.argsort(-rmin)
    s2=s[:n_out]
    sel=xy[:,s2]
    min_t=fluxes.time.min()
    fluxes_m=fluxes[(fluxes.time==min_t)]
    print(fluxes_m.shape)
    sel=pd.DataFrame(sel.T,columns=['x','y'])
    sel['original_index']=np.nan
    sel['User']='extreme'
    for i in range(sel.shape[0]):
        sel.original_index.iloc[i]=int(fluxes_m[(sel.x.iloc[i]==fluxes_m.x) & (sel.y.iloc[i]==fluxes_m.y)].index[0])
    return rmin, xy,sel


#function to get isolate anomaliesnormal without a labelled one in the regions
#needs 2d numpy array with anomalies true false 
def get_isolated_cases(data,labelled,connectivity=2):
    unique_clusters = np.unique((data))
    labeled_clusters = np.zeros_like((data), dtype = np.uint16)
    current_label = 1
    subgroup_labels = {}

    # Initialize the labeled image with valid subgroups 
    filtered_labeled_clusters = np.zeros_like(data, dtype=np.uint16) 


    for cluster_id in unique_clusters:
        cluster_mask = (data == cluster_id).astype(np.uint8)
        # Identify connected components (subgroups)
        subgroups, num_subgroups = label(cluster_mask, connectivity=connectivity, return_num=True) # 2 is diaginal included correct
    
        for subgroup_id in range(1, num_subgroups + 1):
            # Assign a unique label for each subgroup
            labeled_clusters[subgroups == subgroup_id] = current_label
            subgroup_labels[current_label] = f"cluster{cluster_id}_group{subgroup_id}"
            current_label += 1
    #copy the certain in it
    lab2=data.copy()
    for i in range(labelled.shape[0]):
        if labelled.status.iloc[i]=='Anomalous':
            lab2[int(labelled.x.iloc[i]),int(labelled.y.iloc[i])]=1.5
        elif labelled.status.iloc[i]=='Not_anomalous':
            lab2[int(labelled.x.iloc[i]),int(labelled.y.iloc[i])]=-0.5        
    #create x and y array
    xval=lab2.copy()
    yval=lab2.copy()
    for i in range(xval.shape[1]):
        xval[:,i]=range(xval.shape[0])
    for i in range(yval.shape[0]):
        yval[i,:]=range(yval.shape[1])
    list_x=[]
    list_y=[]
    for i in range(np.min(labeled_clusters),np.max(labeled_clusters)+1):
         if np.mean(data[labeled_clusters==i])==0 and np.min(lab2[(labeled_clusters==i)])==0:
             print(f"case {i} of {labeled_clusters[(labeled_clusters==i)].shape[0]} elements")

             if labeled_clusters[(labeled_clusters==i)].shape[0]<=2:
                 list_x.append(xval[(labeled_clusters==i)][0])
                 list_y.append(yval[(labeled_clusters==i)][0])
             else:
                 n=int(labeled_clusters[(labeled_clusters==i)].shape[0]/2)
                 list_x.append(xval[(labeled_clusters==i)][n])
                 list_y.append(yval[(labeled_clusters==i)][n])         
         if np.mean(data[labeled_clusters==i])==1 and np.max(lab2[(labeled_clusters==i)])==1:
             print(f"case {i} of {labeled_clusters[(labeled_clusters==i)].shape[0]} elements")
             if labeled_clusters[(labeled_clusters==i)].shape[0]<=2:
                 list_x.append(xval[(labeled_clusters==i)][0])
                 list_y.append(yval[(labeled_clusters==i)][0])
             else:
                 n=int(labeled_clusters[(labeled_clusters==i)].shape[0]/2)
                 list_x.append(xval[(labeled_clusters==i)][n])
                 list_y.append(yval[(labeled_clusters==i)][n])  
    list_ind=[]
    for i in range(len(list_x)):
        list_ind.append(dat[(dat.x==list_x[i]) & (dat.y==list_y[i])].loc[:,'Unnamed: 0'].iloc[0])

    ar=np.array([list_x,list_y,list_ind])
    df_sel=pd.DataFrame(ar.T,columns=['x', 'y','original_index'])
    print(f"are {df_sel.shape[0]} isolated regions")
    df_sel[['x', 'y','original_index']] = df_sel[['x', 'y','original_index']].astype(int)
    return df_sel
