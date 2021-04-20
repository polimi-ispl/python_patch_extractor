# Patch Extractor

[![Build Status](https://travis-ci.org/polimi-ispl/python_patch_extractor.svg?branch=master)](https://travis-ci.org/polimi-ispl/python_patch_extractor)

A simple yet powerful N-dimensional patch extractor, in pure python!

Authors: [Nicolò Bonettini](mailto:nicolo.bonettini@polimi.it), [Luca Bondi](mailto:luca.bondi@polimi.it), [Francesco Picetti](mailto:francesco.picetti@polimi.it)

## Requirements and Installation
To use this Patch Extractor, simply clone this repo in your work directory.

Actually you will need only `PatchExtractor.py`.

The extractor is built upon `numpy` and `scikit-image` modules. 

## Usage Instructions

1. First you need to import the extractor class:
    ```python
    from python_patch_extractor import PatchExtractor as PE
    ```

2. Then, instantiate an object:
    ```python
    pe = PE.PatchExtractor(dim=patch_shape)
    ```
    `PatchExtractor` class accepts a number of parameters:
     - `dim`: tuple - the shape of every single patch [required].
     - `offset`: tuple - offset to be applied during extraction [None].
     - `stride`: tuple - step length (in samples) between two adjacent patches [None, i.e. equal to `dim`].
     - `rand`: bool - random shuffling the extracted patches [None]. It is concurrent with `function` argument.
     - `function`: function - score function for ordering the patches [None]. It requires the `threshold`argument and it is concurrent with `rand` argument.
     - `threshold`: float - threshold value to be applied to scores computed by `function` [None]. Only the patches that have a score ≥ threshold are returned.
     - `num`: int - number of extracted patches to return [None]. It is concurrent with `indexes` argument.
     - `indexes`: list or 1D array - indexes of the extracted patches to return [None]. It is concurrent with `num` argument.
     - `tapering`: str - tapering function applied to the overlapping portion; must be rect, hanning, cosine, cosinesquare [`rect`]. For now it works only for 2D patches.
     - `padding`: str - padding strategy (taken from np.pad). If the patch is bigger than the input, the original content is centered in the resulting patch. If the patch is smaller than the input, it adds some samples at the end of each axis in order to keep all the available data. 

3. Once you have your data vector (e.g., an image), you can extract patches via:
    ```python
    patches = pe.extract(data)
    ```
   `patches` will have a shape that is a tuple made by two contributions:
   first you have the number of patches extracted for each data dimension,
   then the patch dimension. For examples, if your image is 128 x 128 and you extract
   patches of 64 x 64 (with same stride), your patch array shape will be (2, 2, 64, 64).
   The number of extracted patches will be 4.
   
   You can get the shape of the extracted patches array by using `PE.patch_array_shape(in_size, patch_shape, patch_stride)`.
   
   You can get the number of the extracted patches by using `PE.count_patches(in_size, patch_shape, patch_stride)`.
   
4. If you need to reconstruct the whole data from the patches, you can use:
    ```python
    assembled_data = pe.reconstruct(patches)
    ```
   For now it works for data with up to 4 dimensions.
   
   Please note that `patches` must have the shape provided by the `extract` method. 
   If you have ravelled your patch array, you have lost the spatial information of
   how many patches for each data dimension you extracted.
   However, this information is provided by `PE.patch_array_shape(in_size, patch_shape, patch_stride)`.
   
   Depending on the data shape, patch shape and stride you can loose some data elements (at the axes ends).
   Thus the reconstructed data could be smaller that the original data.
