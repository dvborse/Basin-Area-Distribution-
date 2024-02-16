## Finding basin area distribution (BAD) for independent basins of Islands.

1] BAD can be obtained using the list of area values. In our case we obtained the list of areas from the shapefile of delineated basins. .dbf file can be read with the provided code 
BAD and calculating the power law exponent using MLE. A sample Hawaii shapefile containing .dbf file is also added.

2] The FD_boundary code calculates the fractal dimention of boundary. Here .tiff image can be used as input. Any other format will also work with slight modifications to read the image.
Sample Hawaii.tiff is added.

3] The probabilistic Network evolution model sample code is given here. For more details refer to original article : (https://doi.org/10.1016/j.advwatres.2022.104342) and corresponsing code respository
(https://github.com/dvborse/probabilistic_drainage_network_evolution_model)

