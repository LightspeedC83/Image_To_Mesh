import numpy as np
from PIL import Image
from collections import deque
import math

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

points_by_group = []
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
                    group.append(upNext)
                    visited_spots[upNext[1],upNext[0]] = True #marking this pixel as visited
                    for next_point in [[upNext[0]-1, upNext[1]], [upNext[0]+1, upNext[1]], [upNext[0], upNext[1]-1], [upNext[0], upNext[1]+1]]:
                        if next_point[0]>=0 and next_point[1]>=0 and next_point[0]<x_size and next_point[1]<y_size: #if location is valid 
                            if visited_spots[next_point[1]][next_point[0]] == 0 and pixel_array[next_point[1]][next_point[0]]: #if the pixel at the potential neighbor position isn't already visited and has a black pixel, we add to queue to check
                                queue.append((next_point[0],next_point[1])) 
                group_index +=1
                points_by_group.append(group)



# going through and marking the boundary pixels for each pixel group
boundary_pixles = np.array([[0 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool)
for y in range(reference_size[1]):
    for x in range(reference_size[0]):
        
        for next_point in [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]: 
            if next_point[0]>=0 and next_point[1]>=0 and next_point[0]<x_size and next_point[1]<y_size: #if location is valid                          
                if not pixel_array[next_point[1]][next_point[0]] and groupID_array[y][x]!=None: #if the pixel at the potential neighbor position is white (remember True means black), we mark curr as a border pixel (curr also has to be part of a group)
                    boundary_pixles[y][x] = True
            elif groupID_array[y][x]!=None: #if a pixel that is part of a group is on a the border of the image, it's marked as a border pixel as well
                boundary_pixles[y][x] = True

boundary_points_by_group = []
for g in points_by_group:
    boundary = []
    for p in g:
        if boundary_pixles[p[1]][p[0]]:
            boundary.append(p)
    boundary_points_by_group.append(boundary)



# Now we have the following datastrucutres to keep track information about a pixel at a location (x,y)
#   pixel_array - 2D array pixel_array[y][x] yeilds True if there is a black pixel at (x,y), False if not
#   groupID_array - 2D array pixel_array[y][x] yeilds yeilds the group ID of a pixel at (x,y), None if there is not black pixel there
#   boundary_pixels - 2D array boundary_pixels[y][x] yeilds True if there (x,y) has a boundary pixel, False if not
#   points_by_group - a python list that holds groups, each group is a list of coordinate values
#   bounary_points_by_group - a python list that holds lists of boundary pixels for each group



# now what we want to do is reduce the number of pixels in each border group uniformly by a certain percentage

# first we must rearrainge the order of the points in the boundary_points_by_group list such that points that are next to eachother in the image are next to eachother in the list
# to do this we're basically going to be doing a flood fill algorithm on just the border points
sorted_boundary_points = []
for group in boundary_points_by_group:
    sorted_group =[]

    visited_spots = np.array([[1 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool) # a key of visited spots, False(0) means we haven't visited, True(1) means we have. we set all initially to true and then mark false the points in the group (this ensures that we only consider points in the group)
    point_locations = np.array([[0 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool) # a 2D array of point locations, True means there is a point at said location, False means there is not
    for p in group:
        point_locations[p[1]][p[0]] = True
        visited_spots[p[1]][p[0]] = False

    # now we start the flood fill on this group's border
    queue = deque()
    queue.append(group[0]) # starting point will just be the first one in the group, it doesn't really matter where we start
    while len(queue)!=0:
        curr = queue.pop() #pop returns an element from the right side of the queue, because append adds to the right, this is really functioning like a stack (don't yell at me, I know what I'm doing. Why didn't I call it stack? That's a great question, mind your own fucking business, I don't want to change it)
        sorted_group.append(curr) #add it to the next one
        visited_spots[curr[1]][curr[0]] = True
        for next_point in [[curr[0]-1, curr[1]], [curr[0]+1, curr[1]], [curr[0], curr[1]-1], [curr[0], curr[1]+1]]: #looking at the locations directly adjacent first
            if next_point[0]>=0 and next_point[1]>=0 and next_point[0]<x_size and next_point[1]<y_size: #if location is valid
                if not visited_spots[next_point[1]][next_point[0]]: #if hasn't been visited
                    queue.append(next_point)
                    visited_spots[next_point[1]][next_point[0]] = True
        for next_point in [[curr[0]-1, curr[1]-1], [curr[0]-1, curr[1]+1], [curr[0]+1, curr[1]-1], [curr[0]+1, curr[1]+1]]: # now looking at the locations in the corner
            if next_point[0]>=0 and next_point[1]>=0 and next_point[0]<x_size and next_point[1]<y_size: #if location is valid
                if not visited_spots[next_point[1]][next_point[0]]: #if hasn't been visited
                    queue.append(next_point)
                    visited_spots[next_point[1]][next_point[0]] = True

    sorted_boundary_points.append(sorted_group)

    
reduction_factor = 0.85 #the percentage by which we will reduce the each group list (0.5 reduces the list be one half, 0.25 reduces it by 1/4 (ie. it become 75% of its original size))
reduced_boundaries = []
for group in sorted_boundary_points:
    reduced_group = []
    if reduction_factor <= 0.5 and reduction_factor>=0.01: # lower bound so we don't get absurdly high numbers, negative numbers, or a divide by 0 error
        # TODO: figure out how to max out (adjust beneath) based on density of group for this edge        
        skip_num = round(1/reduction_factor)    
        index = 0
        for p in group: # going through the points sorted by their euclidean distance from the origin
            if index%skip_num != 0: #every skip_num points, we don't add the current point to the list
                reduced_group.append(p)
            index+=1
    else:
        # TODO: figure out how to max out (adjust beneath) based on density of group for this edge    
        skip_num = round(1/(1-reduction_factor))
        index = 0
        for p in group: # going through the points sorted by their euclidean distance from the origin
            if index%skip_num == 0: #every skip_num points, we add the current point to the list
                reduced_group.append(p)
            index+=1

    reduced_boundaries.append(reduced_group)


# removing any empty lists (which may or may not be in here)
i = 0
for g in reduced_boundaries:
    if len(g) == 0:
        del reduced_boundaries[i]
        i-=1
    i+=1


def euclidean_distance(point_one, point_two):
    """calculates euclidean_distance between to points, in the form (int, int)"""   
    return math.sqrt((point_two[0]-point_one[0])**2 + (point_two[1]-point_one[1])**2)

# reduced_boundaries is mostly in order, now we want to identify the points out of place and put them where they should go
for g in range(len(reduced_boundaries)):
    group = reduced_boundaries[g]
    # first we find the average distance between points
    sum = euclidean_distance(group[0], group[-1])
    divisor = 1
    for i in range(len(group)-1):
        sum += euclidean_distance(group[i], group[i+1])
        divisor += 1
    
    avg_distance = sum/divisor
    
    for i in range(len(group)-1):
        
        if euclidean_distance(group[i],group[i+1]) > avg_distance: #if the distance between this point and the next is higher than the benchmark (which at this point is just the avg distance between points for this group)
            #this means that the point at index i+1 is potentially out of place
            best_index = i+1 #start best_index at the index its currently at
            best_left_distance = euclidean_distance(group[i],group[i+1]) #start with current left distance
            try:
                best_right_distance = euclidean_distance(group[i+1], group[i+2]) #start with current right distance
            except IndexError: #in this case i+1 is the end of the list, and we are wrapping around
                best_right_distance = euclidean_distance(group[i+1], group[0])
            
            for j in range(len(group)-1):
                left_distance = euclidean_distance(group[i+1], group[j])
                right_distance = euclidean_distance(group[i+1], group[j+1])
                if left_distance < best_left_distance and right_distance < best_right_distance:
                    best_index = j+1 #use j+1 because if we insert at best_index we want the point at i+1 to be surrounded by the points at j (on left) and j+1 (on right)
                    best_left_distance = left_distance
                    best_right_distance = right_distance
            if euclidean_distance(group[i+1], group[-1]) < best_left_distance and euclidean_distance(group[i+1], group[0]) < best_right_distance: #have to do a check for the loop between the first and last elements
                best_index = 0 
            
            #now we insert the point at i+1 at the best index we've just found (insert first, then remove from where it got shifted to)
            reduced_boundaries[g].insert(best_index, group[best_index])
            del reduced_boundaries[g][i+2]
    


# exporting the point data to a mesh
with open(f"outputs\{reference_name}_output.obj", "w") as out:
    face_index = 1 #vertices are tracked with absolute numbering in order of their definition (for an obj file)
    # first_vertices = [] #for debugging purposes
    # last_vertices = [] #also for debuggin purposes
    # second_vertices = [] #also aslo for dgebugin
    for group in reduced_boundaries: #for each group
        group_vertices = "\n"
        group_faces = "f"
        # first_vertices.append((group[0], face_index))
        for point in group:
            # if first_vertices[-1][1]+1 == face_index:
            #     second_vertices.append((point,face_index))
            group_vertices += f"v {point[0]} {point[1]}\n" #storing this point
            group_faces += f" {face_index}" #adding this vertex to this face list
            face_index +=1 # must increment, and have tracked across groups
        
        # if len(group) != 0:
        #     last_vertices.append((group[-2],face_index))   
      
        out.write(f"\n# group: {groupID_array[group[0][1]][group[0][0]]}")
        out.write(group_vertices)
        out.write(group_faces+"\n")
    
    # for x in last_vertices:
    #     out.write(f"\n#last vertex in group: {groupID_array[x[0][1]][x[0][0]]}\nv {x[0][0]} {x[0][1]} 6 \nf {face_index} {x[1]}")
    #     face_index+=1
    
    # for x in first_vertices:
    #     out.write(f"\n#first vertex in group: {groupID_array[x[0][1]][x[0][0]]}\nv {x[0][0]} {x[0][1]} 2 \nf {face_index} {x[1]}")
    #     face_index+=1
        
    # for x in second_vertices:
    #     out.write(f"\n#second vertex in group: {groupID_array[x[0][1]][x[0][0]]}\nv {x[0][0]} {x[0][1]} 4 \nf {face_index} {x[1]}")
    #     face_index+=1



def save_progression_images():

    # saving an image of all the border pixels marked in red
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


    # saving an image with all the pixels of the same group assigned the same random color
    color_groups ={}
    color_groups[None] = (255,255,255) #white if it's a white pixel

    output_pixels = []
    for y in range(reference_size[1]):
        for x in range(reference_size[0]):
            color = None
            try:
                color = color_groups[groupID_array[y][x]]
            except KeyError:
                color = (np.random.randint(0,250),np.random.randint(0,250),np.random.randint(0,250))
                color_groups[groupID_array[y][x]] = color
            if boundary_pixles[y][x]: #if it's a boundary pixel, we make it black
                color = (0,0,0)
            output_pixels.append(color)

    reference_image = Image.new(mode="RGB", size=reference_size)
    reference_image.putdata(output_pixels)
    reference_image.save(f"images\{reference_name}_groups_marked.png")


    # saving an image of all the border pixels (now reduced) marked in red
    output_pixels = [[(255,255,255) for x in range(x_size)] for y in range(y_size)]
    for g in reduced_boundaries:
        for p in g:
            output_pixels[p[1]][p[0]] = (255,0,0)

    output = []
    for y in output_pixels:
        for x in y:
            output.append(x)

    reference_image = Image.new(mode="RGB", size=reference_size)
    reference_image.putdata(output)
    reference_image.save(f"images\{reference_name}_borders_reduced_by_{int(100*reduction_factor)}_percent.png")    


    # saving an image of the reduced boundary points with all the points of the same group assigned the same random color
    output_pixels = [[(255,255,255) for x in range(x_size)] for y in range(y_size)]
    for g in reduced_boundaries:
        color = (np.random.randint(0,250),np.random.randint(0,250),np.random.randint(0,250))
        for p in g:
            output_pixels[p[1]][p[0]] = color
    out = []
    for y in output_pixels:
        for x in y:
            out.append(x)
    reference_image = Image.new(mode="RGB", size=reference_size)
    reference_image.putdata(out)
    reference_image.save(f"images\{reference_name}_border_groups_marked_reduced_by_{int(100*reduction_factor)}_percent.png")


    #saving the pixels in reduced boundaries, each pixel in the boundary's color is based on it's position in the list
    output_pixels = [[(255,255,255) for x in range(x_size)] for y in range(y_size)]
    for g in reduced_boundaries:
        color_value = 0
        for p in g:
            output_pixels[p[1]][p[0]] = (color_value,0,0)
            color_value += 5
            if color_value >= 255:
                color_value = 255

    output = []
    for y in output_pixels:
        for x in y:
            output.append(x)

    reference_image = Image.new(mode="RGB", size=reference_size)
    reference_image.putdata(output)
    reference_image.save(f"images\{reference_name}_marked_in_order_borders_reduced_by_{int(100*reduction_factor)}_percent.png")

# save_progression_images()