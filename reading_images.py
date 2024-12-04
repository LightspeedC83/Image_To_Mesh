import numpy as np
from PIL import Image
from collections import deque

#opening the reference image
reference_name = "penguin"
reference_path = f"images\{reference_name}.png"

reference_image = Image.open(reference_path)
reference_pixels = list(reference_image.getdata())
reference_size = reference_image.size # getting the refence image's size in (x-pixles,y-pixels) format
x_size = reference_size[0]
y_size = reference_size[1]


# making a greyscale version of the reference image
reference = []
for p in reference_pixels:
    avg = (p[0]+p[1]+p[2])/3
    reference.append((avg,avg,avg))

# take the greyscale image and make it purely black and white
bw_threshold = 50
bw_reference = []
for p in reference:
    if p[0] >= bw_threshold: #if the pixel has value over threshold (ie. pixel is too light), we make it white
        bw_reference.append((255,255,255))
    else: #if value is black enough, we make it black
        bw_reference.append((0,0,0))


# To store the data we want, we'll use a series of 2d Arrays. These will store: bw pixel value, group number, is this pixel an edge pixel for its group
 
#first we create an array for the black-white pixel value, black will be represented as True (1), white as False (0)
pixel_list = [] # a 2D list as specified
temp_list = []
for p in bw_reference: #converting all pixel values in bw_reference to 0s and 1s and storing them in pixel_list accordingly
    if len(temp_list) == reference_size[0]:
        pixel_list.append(temp_list)
        temp_list = []
    
    if p[0] == 0:
        temp_list.append(1)
    else:
        temp_list.append(0)
pixel_list.append(temp_list)

pixel_array = np.array(pixel_list,dtype=bool) # this stores booleans, true means there is a black pixel at said location, false means there is not a black pixel at said location



# now we do a flood fill to assign groupIDs to all pixel objects (white pixels have groupID None)
visited_spots = np.array([[0 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool) # a key of visited spots, False(0) means we haven't visited, True(1) means we have. use this for any given pixel to see if we've visited

# points_by_group = []
groupID_array = np.array([[None for x in range(reference_size[0])]for y in range(reference_size[1])]) #2d array that stores the group id for a pixel at a point, None if it's a white pixel

group_index = 0
for y in range(reference_size[1]):
    for x in range(reference_size[0]):
        curr = pixel_array[y][x] # gives boolean: True if pixel is black, False if not

        if not visited_spots[y][x]: # if the pixel under consideration hasn't been visited

            if not curr: #if location does not have a white we mark as visited and move on
                visited_spots[y][x] = True 
            else:
                queue = deque() #create a queue
                queue.append((x,y)) 
                group = []
                while len(queue)!=0: #while queue isn't empty we keep looking at neighbors of the queue and assigning them IDs of said group
                    upNext = queue.pop()
                    groupID_array[upNext[1]][upNext[0]] = group_index #assigning the group index for this group
                    # group.append(upNext)
                    visited_spots[upNext[1],upNext[0]] = True #marking this pixel as visited
                    for next_point in [[upNext[0]-1, upNext[1]], [upNext[0]+1, upNext[1]], [upNext[0], upNext[1]-1], [upNext[0], upNext[1]+1]]:
                        if next_point[0]>=0 and next_point[1]>=0 and upNext[0]<=x_size and upNext[1]<=y_size: #if location is valid 
                            if visited_spots[next_point[1]][next_point[0]] == 0 and pixel_array[next_point[1]][next_point[0]]: #if the pixel at the potential neighbor position isn't already visited and has a black pixel, we add to queue to check
                                queue.append((next_point[0],next_point[1])) 
                group_index +=1
                # points_by_group.append(group)



# going through and marking the boundary pixels for each pixel group
boundary_pixles = np.array([[0 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool)
for y in range(reference_size[1]-1):
    for x in range(reference_size[0]-1):
        
        for next_point in [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]: 
            if next_point[0]>=0 and next_point[1]>=0 and upNext[0]<=x_size and upNext[1]<=y_size: #if location is valid                          
                if not pixel_array[next_point[1]][next_point[0]] and groupID_array[y][x]!=None: #if the pixel at the potential neighbor position is white (remember True means black), we mark curr as a border pixel (curr also has to be part of a group)
                    boundary_pixles[y][x] = True

# Now we have the following datastrucutres to keep track information about a pixel at a location (x,y)
#   pixel_array - 2D array pixel_array[y][x] yeilds True if there is a black pixel at (x,y), False if not
#   groupID_array - 2D array pixel_array[y][x] yeilds yeilds the group ID of a pixel at (x,y), None if there is not black pixel there
#   boundary_pixels - 2D array boundary_pixels[y][x] yeilds True if there (x,y) has a boundary pixel, False if not


output_pixels = []
for y in range(reference_size[1]):
    for x in range(reference_size[0]):
        
        if boundary_pixles[y][x]:
            output_pixels.append((255,0,0))
        else:
            output_pixels.append((255,255,255))
        

reference_image = Image.new(mode="RGB", size=reference_size)
reference_image.putdata(output_pixels)
reference_image.save(f"images\{reference_name}_borders_marked.png")