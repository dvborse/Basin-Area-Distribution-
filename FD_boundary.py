# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:32:07 2023

@author: Dnyanesh
"""
## Calculate fractal dimension for boundary. Read .tif image of landscape

import cv2
import numpy as np
import matplotlib.pyplot as plt

names=['A&N','auckland','barbados','belle_tie','bermuda','cyprus','domicia','fiji','grenada','haldiboh','hawaii','honolulu','italy','jeju','kauai','mataram','mauritius','moroni','nusa','praia','reunion','taiwan','tasmania','teraciara']

# Read the binary image (replace 'island.png' with your image file path)
#tiff_image = cv2.imread('kauai_fdem.tif', cv2.IMREAD_UNCHANGED)
#binary_image = cv2.imread('fig_square.png', cv2.IMREAD_GRAYSCALE)
#binary_image= cv2.bitwise_not(tiff_image) # inverting manually
#binary_image=tiff_image<204
# Binarize the image (if not already binary)
#_, binary_image_poly = cv2.threshold(tiff_image, 0, 255, cv2.THRESH_BINARY) 
#binary_image_poly=tiff_image>0
FD_values=[]

for name in names:
    
    binary_image_poly=np.load(f'{name}.npy')
    grid_size=binary_image_poly.shape
    shp_boundary=np.zeros((grid_size[0],grid_size[1]))
    
    #All pixel labelled as unassigned "n" & true area matrix defination
    ij=[(0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128)]
    
    for i in range(1,grid_size[0]-1):
        for j in range(1,grid_size[1]-1):
            if binary_image_poly[i,j]>0 :
                for xi in ij:
                    if binary_image_poly[i+xi[0],j+xi[1]]==0:
                        #shp_boundary[i+xi[0],j+xi[1]]=1
                        shp_boundary[i,j]=1
                        
    binary_image=shp_boundary
    
    # Define the range of box sizes (powers of 2)
    box_sizes = 2**np.arange(0, int(np.log2(min(binary_image.shape))) + 1)
    
    # Initialize an array to store box counts
    box_counts = []
    
    # Loop over each box size
    for box_size in box_sizes:
        # Divide the image into boxes of the given size
        rows = binary_image.shape[0] // box_size
        cols = binary_image.shape[1] // box_size
        
        # Reshape the image into boxes
        boxes = binary_image[:rows*box_size, :cols*box_size].reshape(rows, box_size, cols, box_size)
        
        # Count the number of boxes containing object pixels
        box_count = np.sum(np.any(boxes, axis=(1, 3)))
        box_counts.append(box_count)
    
    # Fit a linear regression to calculate the fractal dimension
    coefficients = np.polyfit(np.log(1/box_sizes)[:-3], np.log(box_counts)[:-3], 1)
    fractal_dimension = coefficients[0]
    
    # Plot the results on log-log axes with increased font size
    plt.figure()
    plt.loglog(1/box_sizes, box_counts, 'bo', label='Box Counts', markersize=6)
    plt.plot(1/box_sizes[:-3], np.exp(coefficients[1]) * box_sizes[:-3]**(-coefficients[0]), 'r-', label='Fitting Line', linewidth=2)
    plt.xlabel('1/Box Size', fontsize=14)
    plt.ylabel('Box Counts', fontsize=14)
    plt.title(f'{name} FD boundary Y={coefficients[1]:.2f}*X^{fractal_dimension:.3f}', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"FD_boundary_{name}_{fractal_dimension:.3f}.png", bbox_inches='tight',dpi=300)
    #plt.legend(fontsize=12)
    plt.show()
    plt.close()
    
    # Display the fractal dimension with increased font size
    #plt.annotate(f'Fractal Dimension: {fractal_dimension:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14)
    FD_values.append(fractal_dimension)
    # Display the results
    print(f'Box Counts: {box_counts}')
    print(f'Fractal Dimension: {fractal_dimension:.4f}')
