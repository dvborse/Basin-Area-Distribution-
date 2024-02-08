# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:22:04 2023

@author: dnyan
"""

#Exceedance probability distribution independent basin area distribution
# We are considering top 20% of the basins

import numpy as np
import matplotlib.pyplot as plt
from dbfread import DBF

#Names=['barbados','belle_tie','bermuda','fiji','grenada','italy','jeju','kauai']
Names=['a&n','auckland','cyprus','haldiboh','hawaii','honolulu','port_francie','reunion','taiwan','tasmania']

for name in Names:

    table = DBF(f'{name}.dbf')
    
    # Extract field names from the table
    field_names = table.field_names
    
    # Initialize an empty list to hold records
    records = []
    
    # Loop through the records and append them to the list
    for record in table:
        records.append([record[field] for field in field_names])
    
    # Convert the list of records into a NumPy array
    data = np.array(records)
    
    area_list=list(data[:,2])
    
    # th=1
    # for i in range(grid_size[0]): 
    #     for j in range(grid_size[1]):
    #         if true_area[i,j]:
    #             if Label[(i,j)]=='b':
    #                 if Facc[i][j]>th:
    #                     area_list.append(Facc[i][j])
    
    area_list.sort(reverse=True)
    
    area=np.array(area_list)
    
    #n=int(len(area_list)/5)  # take 25% data
    
    #area=area[:n]        
    area=area[area>=50000]
    
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
    ex = 1 + len(area) / mle
    k = (1 - ex) / (xmax**(1 - ex) - xmin**(1 - ex))
    
    exp = ex - 1
    coe = k / (ex - 1)
    
    P_area_cal = coe * P_area[:, 0]**(-exp)
    
    # Individual plot
    title_str = f'P = {coe:.2f}x^{exp:.2f}'
    filename = f'{name}_th05_{exp:.2f}'
    
    plt.loglog(P_area[:, 0], P_area[:, 1], "o", markersize=8)
    plt.loglog(P_area[:, 0], P_area_cal, "r--", linewidth=2)
    plt.xlabel('Area', fontsize=16)
    plt.ylabel('P[A>A*]', fontsize=16)
    plt.title(f'{filename} Basin_dist {title_str}')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'Basin_dist {filename}.png', bbox_inches='tight', dpi=300)
    #plt.savefig(f'Ex_IA_dist{filename}.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
'''
normalised_area=area_list/np.sum(area_list)

n_points=100 #Let's choose 100 points on logarithmic scale for distribution

max_area=np.amax(normalised_area[:])
min_area=np.amin(normalised_area[:])

P_area=np.zeros((n_points,2)) #first column facc value and second column probability
Total_a=len(normalised_area);
area_range=np.arange(np.log(min_area),np.log(max_area),(np.log(max_area)-np.log(min_area))/n_points)
#area_range=area_range[:-1]

for i in range(len(area_range)):
    temp_a=np.exp(area_range[i])
    P_area[i,0]=temp_a
    P_area[i,1]=np.sum(normalised_area[:]>temp_a)/Total_a
    
plt.loglog((P_area[:,0]),(P_area[:,1]),linewidth=0, color='royalblue',marker='o',markersize=2)  
#choose number of points till slopebreak

lim1=10
lim2=75


for ix in P_area[:,1]:
    if ix>0.005:
        lim2+=1
    else:
        break

 
#with slope break
slope_a, intercept_a = np.polyfit(np.log(P_area[lim1:lim2,0]),np.log(P_area[lim1:lim2,1]), 1)

intercept_a=np.exp(intercept_a)
P_area_cal=intercept_a*P_area[lim1:lim2,0]**slope_a

plt.loglog((P_area[:,0]),(P_area[:,1]),linewidth=0, color='royalblue',marker='o',markersize=2)
plt.loglog((P_area[lim1:lim2,0]),(P_area_cal) ,color='black',linewidth=0.5)
plt.xlabel('Area(\u03B4)',fontsize='12')
plt.ylabel('P[Ad>=\u03B4]',fontsize='12')
plt.title(f"\u03B1={alpha} & \u03B2={beta} th:{th} Y={intercept_a:.2f}*X^{slope_a:.2f}", fontsize=8)
plt.savefig("Ex_prob_dist th:{th}",bbox_inches='tight',dpi=300)
plt.show()
'''
