# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:44:24 2023

@author: Dnyanesh
"""
import imageio
from PIL import Image
import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import matplotlib.colors
import time
import winsound

tic=time.time()

#alpha_values=np.arange(0,1.1,0.1)
alpha_values=np.arange(0,1.1,0.5)
iterations=20 # for each alpha value
name='Hawaii'
th=25 # for basin area distribution
#Area_actual=10462.94/0.0009   # number of cells from area in km2 (km2) for using same th 0.05km2

# #image = Image.open('hawaii_polygon_0.005.tif')
# image_np = np.array(np.flipud(image[:,:,0]))


tiff_image = cv2.imread(f'{name}.tif', cv2.IMREAD_UNCHANGED)
#temp=tiff_image>0
#temp=np.flipud(tiff_image[:,:,0]>0)
temp=tiff_image==330
grid_size=temp.shape
binary_image_poly=np.zeros((grid_size[0]+2,grid_size[1]+2))
binary_image_poly[1:-1,1:-1]=temp

arr_shp=binary_image_poly

grid_size=arr_shp.shape
arr_boundary = np.zeros((grid_size[0],grid_size[1]))

beta=1 # constant for all
#alpha=1

def FD_cordinates(xy):
    
    if FD[xy[0]][xy[1]]==1:
        return((xy[0],xy[1]+1))
    if FD[xy[0]][xy[1]]==2:
        return((xy[0]+1,xy[1]+1))
    if FD[xy[0]][xy[1]]==4:
        return((xy[0]+1,xy[1]))
    if FD[xy[0]][xy[1]]==8:
        return((xy[0]+1,xy[1]-1))
    if FD[xy[0]][xy[1]]==16:
        return((xy[0],xy[1]-1))
    if FD[xy[0]][xy[1]]==32:
        return((xy[0]-1,xy[1]-1))
    if FD[xy[0]][xy[1]]==64:
        return((xy[0]-1,xy[1]))
    if FD[xy[0]][xy[1]]==128:
        return((xy[0]-1,xy[1]+1))

def Get_cord(ix):
    i=int(ix/grid_size[1])
    j=ix%grid_size[1]
    return((i,j))

def Get_id(i,j):
    return grid_size[1]*i+j

norm=plt.Normalize(0,1)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","blue"])

exp_values=np.zeros((len(alpha_values),iterations,2))

#%% Probabilistic Model

for it in range(len(alpha_values)):
    for itt in range(iterations):
        alpha=alpha_values[it]
    
        # MODEL
        #flow direction and flow accumulation matrices
        FD=np.zeros((grid_size[0],grid_size[1]))
        Facc=np.zeros((grid_size[0],grid_size[1]))    
        Length=np.zeros(grid_size[0]*grid_size[1])      # single array
        Down_length=np.zeros((grid_size[0],grid_size[1])) 
        Pot_ids=[]
        
        Label=dict()
        
        ## Initial Setup
        
        true_area=arr_shp>0 # matrix true for area and boundary
        
        #All pixel labelled as unassigned "n" & true area matrix defination
        ij=[(0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128)]
        
        for i in range(1,grid_size[0]-1):
            for j in range(1,grid_size[1]-1):
                if true_area[i,j]>0 :
                    Label.update({(i,j):"n"})
                    for xi in ij:
                        if arr_shp[i+xi[0],j+xi[1]]==0:
                            arr_boundary[i+xi[0],j+xi[1]]=1
                            Label.update({(i,j):"b"})
                            Facc[i,j]=1
        
        # Append potential 
        for pi in Label:
            if Label[(pi[0],pi[1])]=='b':
                for y in ij:
                    if arr_shp[(pi[0]+y[0],pi[1]+y[1])]>0:
                        if Label[(pi[0]+y[0],pi[1]+y[1])]=='n':
                            iid=Get_id(pi[0]+y[0],pi[1]+y[1])
                            Pot_ids.append(iid)
                            Label.update({(pi[0]+y[0],pi[1]+y[1]):"p"})
                            Length[iid]=1
                            #Down_length[pi[0]+y[0],pi[1]+y[1]]=1
        
        
        while(len(Pot_ids)>0):
            
            #Randomly choosing one pixel from Pot_ids list
            #choosing one pixel from Pot_ids list based on probability given by Down_length
            pot_len=Length[Pot_ids]
            len_alpha=pot_len**alpha
            cum_len=np.cumsum(len_alpha)
            
            rand_n=random.randint(1,int(cum_len[-1]))
            
            pi_id=-1
            for j in range(len(Pot_ids)):
                if rand_n<=cum_len[j]:
                    pi_id=Pot_ids[j]
                    break
                
            pi=Get_cord(pi_id)
            
            ## Flow direction using beta
            Values=[] #list of tupules containing FD, Facc and Facc cumulative for surrounding drainage pixels
            Facc_cum=0
            
            for y in ij:
                if true_area[pi[0]+y[0],pi[1]+y[1]]: 
                    if Label[(pi[0]+y[0],pi[1]+y[1])]=='y' or Label[(pi[0]+y[0],pi[1]+y[1])]=='b':
                          Facc_cum+=int((Facc[pi[0]+y[0]][pi[1]+y[1]])**beta)
                          Values.append((y[2],Facc[pi[0]+y[0]][pi[1]+y[1]],Facc_cum))
        
            sum_facc=Values[-1][2]
            value_rand=random.randint(1,sum_facc)
            
            for i in range(len(Values)):
                if value_rand<=Values[i][2]:
                    FD[pi[0]][pi[1]]=Values[i][0]
                    break
            
            #Now updating labels and list for pi 
            Pot_ids.remove((pi_id))
            Label.update({pi:"y"})
            
            #Updating Down_length for pi
            cordinates=FD_cordinates(pi)
            if (FD[pi[0]][pi[1]]==2 or FD[pi[0]][pi[1]]==8 or FD[pi[0]][pi[1]]==32 or FD[pi[0]][pi[1]]==128):
                Down_length[pi[0]][pi[1]]=Down_length[cordinates[0]][cordinates[1]]+np.sqrt(2)
            else:
                Down_length[pi[0]][pi[1]]=Down_length[cordinates[0]][cordinates[1]]+1
              
            #Updating labels and lists for and surrounding pixels of pi
            for x in ij:
                if Label[(pi[0]+x[0],pi[1]+x[1])]=='n':
                    Label[(pi[0]+x[0],pi[1]+x[1])]='p'
                    Pot_ids.append(grid_size[1]*(pi[0]+x[0])+pi[1]+x[1])
                    Down_length[pi[0]+x[0]][pi[1]+x[1]]=1+Down_length[pi[0]][pi[1]]
                    Length[grid_size[1]*(pi[0]+x[0])+pi[1]+x[1]]=1+Length[grid_size[1]*pi[0]+pi[1]]
                
            #Now updating flow accumulation value for pi and then along the stream if flow direction is updated or keep it same
        
            current_pi=(pi[0],pi[1])
            Facc[pi[0]][pi[1]]=1
            
            while(not(Label[(current_pi[0],current_pi[1])]=="b")):
                current_pi=FD_cordinates(current_pi)
                Facc[current_pi[0]][current_pi[1]]+=1  
        
        if itt==0:
            W=np.where(Facc>50,1,0)
            plt.imshow(W, cmap=cmap)
            plt.title(f"Network alpha={alpha:.1f}", fontsize=8)
            plt.savefig(f"Network alpha={alpha:.1f} beta={beta}.png", bbox_inches='tight',dpi=300)
            plt.close()
        
        #%% Basin Distribution
        area_list=[]
        #th=25
        #Traversing trough boundary pixels
        for bp in Label: 
            if Label[bp]=="b": 
                if Facc[bp[0],bp[1]]>th:
                    area_list.append(Facc[bp[0],bp[1]])
    
        area_list.sort(reverse=True)
        area_list=np.array(area_list)
    
        area=area_list[area_list>=th]
        area_range = np.unique(area)
    
        total=len(area)
        length=len(area_range)
    
        P_area=np.zeros((length,2))
    
        for i in range(len(area_range)):
            P_area[i,0]=area_range[i];
            P_area[i,1]=sum(area[:]>=area_range[i])/total;
    
        # Exponent and coefficient with MLE
        # MLE
        xmin = np.min(area)
        xmax = np.max(area)
        Area = area / xmin
        mle = np.sum(np.log(area / xmin))
        alpha1 = 1 + len(area) / mle
        k = (1 - alpha1) / (xmax**(1 - alpha1) - xmin**(1 - alpha1))
    
        exp = alpha1 - 1
        coe = k / (alpha1 - 1)
    
        P_area_cal = coe * P_area[:, 0]**(-exp)
        
        exp_values[it,itt,0]=exp
        exp_values[it,itt,1]=coe
        
        if itt==0:
            # Individual plot
            title_str = f'P = {coe:.2f}x^{exp:.2f}'
            filename = f'{name}_{alpha:.1f}_basin_area_dist {th}'
        
            plt.loglog(P_area[:, 0], P_area[:, 1], "o", markersize=8)
            plt.loglog(P_area[:, 0], P_area_cal, "r--", linewidth=2)
            plt.xlabel('Area', fontsize=16)
            plt.ylabel('P[A>A*]', fontsize=16)
            plt.title(f'{name}_{alpha:.1f} th_{th} BAD {title_str}')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(f'{filename}_{exp:.2f}.png', bbox_inches='tight', dpi=300)
            plt.savefig(f'{filename}_{exp:.2f}.svg', bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()
            #%% Basin Boundaries
            basins=np.zeros((grid_size[0],grid_size[1]))
        
            #xij=[(0,1,16),(1,1,32),(1,0,64),(1,-1,128),(0,-1,1),(-1,-1,2),(-1,0,4),(-1,1,8)]
            ij=[(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        
            def if_FD(x_pi,value):
                # Give input list of pixel, function will check if surrounding pixels are flowing towards it
                # If it's flowing towards then it will add it to watershed and move on 
                # value will be whatever value being assigned to that watershed
                for x in ij:
                    if true_area[x_pi[0]+x[0],x_pi[1]+x[1]]:    
                        if FD_cordinates((x_pi[0]+x[0],x_pi[1]+x[1]))==x_pi:
                            basins[x_pi[0]+x[0],x_pi[1]+x[1]]=value
                            next_pi=(x_pi[0]+x[0],x_pi[1]+x[1])
                            if_FD(next_pi,value)
                    
        
            # Outlets list
            outlets=[]
            for i in range(grid_size[0]-2):
                for j in range(grid_size[1]-2):
                    if true_area[i+1,j+1]:
                        if Label[(i+1,j+1)]=='b':
                            outlets.append((i+1,j+1))
                        
            for i in range(len(outlets)):
                xpi=outlets[i]
                if_FD(xpi,i+1)
                
            # Basin_bondaries in ratser form
            basin_boundaries=np.zeros(((grid_size[0],grid_size[1])))
        
            for i in range(1,grid_size[0]-1):
                for j in range(1,grid_size[1]-1):
                    if true_area[i,j]:
                        for x in ij:
                            if true_area[i+x[0],j+x[1]]:
                                if basins[i,j]!=basins[i+x[0],j+x[1]]:
                                    if basin_boundaries[i+x[0],j+x[1]]==0:
                                        basin_boundaries[i,j]=1
                                        break
            
            cmap2= matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","black"])
        
            basin_boundaries=np.flipud(basin_boundaries)
            plt.imshow(basin_boundaries, cmap=cmap2, origin='lower',extent=[0,grid_size[1],0,grid_size[0]])
            #plt.title('stream Network for iteration '+str(ixx), fontsize=8)
            plt.savefig(f"{name}_basin_boundaries \u03B1={alpha:.1f}.svg", bbox_inches='tight',dpi=300)
            plt.savefig(f"{name}_basin_boundaries \u03B1={alpha:.1f}.png", bbox_inches='tight',dpi=300)
            plt.close()
        
#%% Calculation of GC
'''
import cv2

# Read the binary image (replace 'island.png' with your image file path)
#binary_image = cv2.imread(f'{name}.png', cv2.IMREAD_GRAYSCALE)
binary_image = cv2.imread(f'{name}.tif', cv2.IMREAD_GRAYSCALE)


binary_image= cv2.bitwise_not(binary_image) # inverting manually

# Binarize the image (if not already binary)
_, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

# Calculate the perimeter using findContours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
perimeter = sum(cv2.arcLength(cnt, closed=True) for cnt in contours)

# Calculate the area using contour area
area = sum(cv2.contourArea(cnt) for cnt in contours)

# Calculate the radius of a circle with the same area
radius = np.sqrt(area / np.pi)

# Calculate the circumference of the circle
circumference = 2 * np.pi * radius

# Calculate the Gravelius Compactness Coefficient
GC= perimeter / circumference

# Display the results
print(f'Perimeter: {perimeter}')
print(f'Area: {area}')
print(f'Gravelius Coefficient: {GC:.2f}')

#%% FD Boundary 

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
# Display the results
print(f'Box Counts: {box_counts}')
print(f'Fractal Dimension: {fractal_dimension:.4f}')
#%% FD calculation Box

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
coefficients = np.polyfit(np.log(1/box_sizes), np.log(box_counts), 1)
fractal_dimension = coefficients[0]

# Plot the results on log-log axes with increased font size
plt.figure()
plt.loglog(1/box_sizes, box_counts, 'bo', label='Box Counts', markersize=6)
plt.plot(1/box_sizes, np.exp(coefficients[1]) * box_sizes**(-coefficients[0]), 'r-', label='Fitting Line', linewidth=2)
plt.xlabel('1/Box Size', fontsize=14)
plt.ylabel('Box Counts', fontsize=14)
plt.title(f'{name}_FD', fontsize=14)
#plt.legend(fontsize=12)

# Display the fractal dimension with increased font size
plt.annotate(f'Fractal Dimension: {fractal_dimension:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14)

# Set tick label font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"{name}_FD_{fractal_dimension:.2f}.png", bbox_inches='tight',dpi=300)
plt.show()

# Display the results
print(f'Box Counts: {box_counts}')
print(f'Fractal Dimension: {fractal_dimension:.2f}')
'''