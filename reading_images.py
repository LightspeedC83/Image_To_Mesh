import numpy as np
from PIL import Image
from collections import deque
import math
import shapely.geometry as s
import anytree

#opening the reference image
reference_name = "complex_test_small"
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


print("doing flood fill...")
# now we do a flood fill to assign groupIDs to all pixel objects (white pixels have groupID None)
visited_spots = np.array([[0 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool) # a key of visited spots, False(0) means we haven't visited, True(1) means we have. use this for any given pixel to see if we've visited

points_by_group = []
groupID_array = np.array([[None for x in range(reference_size[0])]for y in range(reference_size[1])]) #2d array that stores the group id for a pixel at a point, None if it's a white pixel

group_index = 0
for y in range(reference_size[1]):
    for x in range(reference_size[0]):
        curr = pixel_array[y][x] # gives boolean: True if pixel is black, False if not

        if not visited_spots[y][x]: # if the pixel under consideration hasn't been visited

            if not curr: #if location does not have a black pixel we mark as visited and move on
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
                            if not visited_spots[next_point[1]][next_point[0]] and pixel_array[next_point[1]][next_point[0]]: #if the pixel at the potential neighbor position isn't already visited and has a black pixel, we add to queue to check
                                queue.append((next_point[0],next_point[1])) 
                group_index +=1
                points_by_group.append(group)


print("identifying boundaries...")
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
#TODO: boundary_points_by_group has way more points than it should, figure out what the fuck is happening!!!! Probably what's going to need to happen is rewriting the whole boundary (and/or group) identification program :( :( :( :( :(

# Now we have the following datastrucutres to keep track information about a pixel at a location (x,y)
#   pixel_array - 2D array pixel_array[y][x] yeilds True if there is a black pixel at (x,y), False if not
#   groupID_array - 2D array pixel_array[y][x] yeilds yeilds the group ID of a pixel at (x,y), None if there is not black pixel there
#   boundary_pixels - 2D array boundary_pixels[y][x] yeilds True if there (x,y) has a boundary pixel, False if not
#   points_by_group - a python list that holds groups, each group is a list of coordinate values
#   bounary_points_by_group - a python list that holds lists of boundary pixels for each group



# now what we want to do is reduce the number of pixels in each border group uniformly by a certain percentage

print("doing flood fill on boundary points...")
# first we must rearrainge the order of the points in the boundary_points_by_group list such that points that are next to eachother in the image are next to eachother in the list
# to do this we're basically going to be doing a flood fill algorithm on just the border points
sorted_boundary_points = []
for group in boundary_points_by_group:
    
    
    visited_spots = np.array([[1 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool) # a key of visited spots, False(0) means we haven't visited, True(1) means we have. we set all initially to true and then mark false the points in the group (this ensures that we only consider points in the group)
    point_locations = np.array([[0 for x in range(reference_size[0])]for y in range(reference_size[1])], dtype=bool) # a 2D array of point locations, True means there is a point at said location, False means there is not
    for p in group:
        point_locations[p[1]][p[0]] = True
        visited_spots[p[1]][p[0]] = False


    #create a copy of the group list with no duplicates
    group_copy = []
    seen_points = {}
    for point in group:      
        if seen_points.get((point[1],point[0])) == None: #if we haven't seen the point we're looking at, we add it to new group list and mark it as visited
            group_copy.append(point)
            seen_points[(point[1],point[0])] = True
    
    
    while len(group_copy) > 0: # This loop makes it such that it will do the flood fill on everything in group, so if there are isolated points, those will get their own list in sorted_boundary_points
        sorted_group = []

        # now we start the flood fill on this group's border
        queue = deque()
        queue.append(group_copy[0]) # starting point will just be the first one in the group, it doesn't really matter where we start
        while len(queue)!=0:
            curr = tuple(queue.pop()) #pop returns an element from the right side of the queue, because append adds to the right, this is really functioning like a stack (don't yell at me, I know what I'm doing. Why didn't I call it stack? That's a great question, mind your own fucking business, I don't want to change it)
            group_copy.remove(curr)
            sorted_group.append(curr) #add it to the next one
            visited_spots[curr[1]][curr[0]] = True
            for next_point in [[curr[0]-1, curr[1]], [curr[0]+1, curr[1]], [curr[0], curr[1]-1], [curr[0], curr[1]+1]]: #looking at the locations directly adjacent first
                if next_point[0]>=0 and next_point[1]>=0 and next_point[0]<x_size and next_point[1]<y_size: #if location is valid
                    if not visited_spots[next_point[1]][next_point[0]]: #if hasn't been visited
                        queue.append(next_point)
                        visited_spots[next_point[1]][next_point[0]] = True
            for next_point in [[curr[0]-1, curr[1]-1], [curr[0]-1, curr[1]+1], [curr[0]+1, curr[1]-1], [curr[0]+1, curr[1]+1]]: # now looking at the locations in the corners
                if next_point[0]>=0 and next_point[1]>=0 and next_point[0]<x_size and next_point[1]<y_size: #if location is valid
                    if not visited_spots[next_point[1]][next_point[0]]: #if hasn't been visited
                        queue.append(next_point)
                        visited_spots[next_point[1]][next_point[0]] = True

        sorted_boundary_points.append(sorted_group)


 # saving an image with all the points of the same group assigned the same random color
output_pixels = [[(255,255,255) for x in range(x_size)] for y in range(y_size)]

for g in sorted_boundary_points:
    color = (np.random.randint(0,250),np.random.randint(0,250),np.random.randint(0,250))
    for p in g:
        output_pixels[p[1]][p[0]] = color
    
out = []
for y in output_pixels:
    for x in y:
        out.append(x)
reference_image = Image.new(mode="RGB", size=reference_size)
reference_image.putdata(out)
reference_image.save(f"images\{reference_name}_testing_border_sort.png")


print("reducing the boundaries...")
reduction_factor = 0.5 #the percentage by which we will reduce the each group list (0.5 reduces the list be one half, 0.25 reduces it by 1/4 (ie. it become 75% of its original size))
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


# removing any empty lists (which may or may not be in here) as well as any groups that have 2 or fewer vertices (as you need 3 vertices to make a face)
i = 0
for g in reduced_boundaries:
    if len(g) <= 2:
        del reduced_boundaries[i]
        i-=1
    i+=1


print("reloacting outliers in the boundary...")

def euclidean_distance(point_one, point_two):
    """calculates euclidean_distance between to points, in the form (int, int)"""   
    return math.sqrt((point_two[0]-point_one[0])**2 + (point_two[1]-point_one[1])**2)

# reduced_boundaries is mostly in order, now we want to identify the points out of place and put them where they should go, we make a function that does this on a generic list of the same structure as reduced_boundaries as it will be useful later
def relocate_points(boundary_points_input):
    """function that will, given an input list in the form [[(x1,y1),(x2,y2), ...], [(x3,y3),(x4,y4), ...], ...], 
    will identify outlier points and relocate them in the list to their proper position such that points that are 
    next to eachother on the boundary are next to eachother in the list. This function doesn't return anything, 
    it just modifies the list."""

    for g in range(len(boundary_points_input)):
        group = boundary_points_input[g]
        # first we find the average distance between adjacent points
        sum = euclidean_distance(group[0], group[-1])
        divisor = 1
        for i in range(len(group)-1):
            sum += euclidean_distance(group[i], group[i+1])
            divisor += 1
        
        avg_distance = sum/divisor
        
        for i in range(len(group)-1):
            #commenting out the check for if it's over average distance and doing relocation on every point
            # if euclidean_distance(group[i],group[i+1]) > avg_distance: #if the distance between this point and the next is higher than the benchmark (which at this point is just the avg distance between points for this group)
            # # this means that the point at index i+1 is potentially out of place
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
            boundary_points_input[g].insert(best_index, group[best_index])
            del boundary_points_input[g][i+2]

relocate_points(reduced_boundaries)

# removing any duplicate points within groups (i don't want to just convert to a set and convert back because order may or may not be preserved, why didn't i put this bit of code back when I got rid of empty groups? that's a great question, I origionally had done exactly that, but after doing the relocation of outliers step, i'd get duplicate points soooooo I'm putting this bit of code here...)
seen_points = {}
reduced_boundaries_removed_duplicates = []
for group in reduced_boundaries:
    non_duplicates = []
    for point in group:
        if seen_points.get((point[1],point[0])) == None: #if we haven't seen the point we're looking at, we add it to new group list and mark it as visited
            non_duplicates.append(point)
            seen_points[(point[1],point[0])] = True
    reduced_boundaries_removed_duplicates.append(non_duplicates)
reduced_boundaries = reduced_boundaries_removed_duplicates


# Now we need to determine whether or not any of the groups are inside any other groups, any group inside of another gets put in a hierarchical structure, where it's subordinate to the group it's inside
class Boundary:
    """keeps track of boundary points and the polygon object created by those points"""
    def __init__(self, points, id):
        self.points = points
        self.polygon = s.Polygon(points)
        self.id = id

group_trees = []
id = 0
for group in reduced_boundaries:
    group_trees.append(anytree.Node(f"boundary {id}", boundary=Boundary(group, id)))
    id+=1

# going through the boudary objects (all starting in tree roots) and if a boundary has a point inside another boundary, we subordinate that point to it in the tree structure
ids_subordinated = []
for i in range(len(group_trees)):
    curr = group_trees[i]
    for j in range(len(group_trees)):
        if i != j and group_trees[j].boundary.polygon.contains(s.Point(curr.boundary.points[0])):  #if the current node we're looking at is contained inside another's boundary
            #we need to insert at the lowest possible point in the hierarchy
            if len(group_trees[j].children) > 0: # if the node we are going to insert at has children
                queue = deque()
                for x in group_trees[j].children: #putting the node and the depth from the starting node to be stored in the queue
                    queue.append((x,1))
                best = (group_trees[j], 0)
                
                while len(queue)!=0:
                    upNext = queue.pop()
                    child = upNext[0]
                    #check to see if the current boundary is inside the child's polygon
                    if child.boundary.polygon.contains(s.Point(curr.boundary.points[0])) and upNext[1] > best[1]: # if the child contains curr and it's depth in the tree is greater than the current best depth
                        best = upNext

                    for candidate in child.children:
                        queue.append((candidate, upNext[1]))

                # now that we've found the node that contains curr that's deepest in the tree
                curr.parent = best[0]
                if len(best[0].children) > 0: # if the new parent of curr has children, we make curr their parent, if they fit inside curr
                    for child in best[0].children:
                        if curr.boundary.polygon.contains(s.Point(best[0].boundary.points[0])):
                            child.parent = curr
                

            else: # if the node we want to insert at doesn't have children we just make curr it's child and move on with our life
                curr.parent = group_trees[j]
            ids_subordinated.append(curr.boundary.id)


# removing from the main list any boundary groups that were subordinated to another group, we only want to hold root nodes in this list
for tree in group_trees:
    if tree.boundary.id in ids_subordinated:
        group_trees = [x for x in group_trees if x!=tree]

for t in group_trees:
    print(anytree.RenderTree(t))






print("writing to an .obj file...")

# exporting the point data to a mesh
extrusion = 5 #how much extrusion the mesh should be given in the z direction
open_backed = False
create_offset_socket = False
offset_scalar = 2
offset_point_distance_proportion = 0.35

obj_vertex_numbers = {}
with open(f"outputs\{reference_name}_output.obj", "w") as out:
    vertex_index = 1 #vertices are tracked with absolute numbering in order of their definition (for an obj file)
    # creating the vertices and the n-gon faces based on the image for each group
    for group in reduced_boundaries: 
        group_vertices = "\n"
        group_faces = "f"
        
        for point in group:
            group_vertices += f"v {point[0]} {point[1]} 0\n" #storing this point
            group_faces += f" {vertex_index}" #adding this vertex to this face list
            obj_vertex_numbers[(point[0], point[1], 0)] = vertex_index #keeping track of this vertex's index
            vertex_index +=1 # must increment, and have tracked across groups

        out.write(f"\n# group: {groupID_array[group[0][1]][group[0][0]]}")
        out.write(group_vertices)
        out.write(group_faces+"\n")
    
    if extrusion != 0:
        # creating the vertices and faces for the extruded
        for group in reduced_boundaries:
            extruded_vertices = "\n"

            # saving all the vertices offset on the z-axis by the extrusion value
            for point in group:
                extruded_vertices += f"v {point[0]} {point[1]} {extrusion}\n"
                obj_vertex_numbers[(point[0], point[1], extrusion)] = vertex_index
                vertex_index +=1

            out.write(f"\n# Extruded vertices for group: {groupID_array[group[0][1]][group[0][0]]}")
            out.write(extruded_vertices)

            for i in range(len(group)-1):
                out.write(f"f {obj_vertex_numbers[(group[i][0], group[i][1], 0)]} {obj_vertex_numbers[(group[i+1][0], group[i+1][1], 0)]} {obj_vertex_numbers[(group[i+1][0], group[i+1][1], extrusion)]} {obj_vertex_numbers[(group[i][0], group[i][1], extrusion)]}\n")
            #have to include link between last and first points in the list
            out.write(f"f {obj_vertex_numbers[(group[-1][0], group[-1][1], 0)]} {obj_vertex_numbers[(group[0][0], group[0][1], 0)]} {obj_vertex_numbers[(group[0][0], group[0][1], extrusion)]} {obj_vertex_numbers[(group[-1][0], group[-1][1], extrusion)]}\n")

        if not open_backed: # closes the back of the extruded part with an n-gon
            for group in reduced_boundaries:
                face = "f"
                for point in group:
                    face += f" {obj_vertex_numbers[(point[0],point[1], extrusion)]}"
                face += f" {obj_vertex_numbers[(group[0][0],group[0][1], extrusion)]}"
                out.write(f"\n# Back face for group: {groupID_array[group[0][1]][group[0][0]]}\n")
                out.write(face)
    
    ## offset boundary (ie. expanded boundary)

    if create_offset_socket and offset_scalar != 0:
        offset_boundary_points = []
        for group in reduced_boundaries:
            
            polygon = s.Polygon(group) # we'll use the shapely.geometry library to figure out if a candidate offset point produces a collision with the shape
            offset_group = []

            for i in range(len(group)-1):
                #getting the possible boundary points on one side of the boundary
                candidates_side_one = [] #contains two points on one side of the boundary curve
                #getting the point from the normal vector between point at i-1 and point at i
                n = (group[i-1][1]-group[i][1], -1*(group[i-1][0]-group[i][0])) # if i=0, then this will reach back around to the point at the end of the list, meaning group[i-1] will be group[-1] (Note: it's very important that the -1* component is different when looking back to when looking forward)
                n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
                n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
                candidates_side_one.append((group[i][0]+offset_scalar*n_unit[0], group[i][1]+offset_scalar*n_unit[1])) #TODO: consider making a function that calculates and returns point based on two inputted points and an offset?

                #getting point from normal vector between point at i and point at i+1
                n = (-1*(group[i+1][1]-group[i][1]), group[i+1][0]-group[i][0]) #get normal vector by flipping x & y components of the vector between the point at i and the point at i+1 and negating one of those componenets, which component you negate changes which normal vector is used
                n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
                n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
                # only add the point from this normal vector if it's not too close to the point from the previous normal vector (determined by offset_point_distance_proportion and offset_scalar)
                if euclidean_distance(candidates_side_one[0], (group[i][0]+offset_scalar*n_unit[0], group[i][1]+offset_scalar*n_unit[1])) > offset_point_distance_proportion*offset_scalar:
                    candidates_side_one.append((group[i][0]+offset_scalar*n_unit[0], group[i][1]+offset_scalar*n_unit[1]))

                #getting the possible boundary points on the other side of the boundary
                candidates_side_two = [] #contains two points on one side of the boundary curve
                #getting the point from the normal vector between point at i-1 and point at i
                n = (-1*(group[i-1][1]-group[i][1]), group[i-1][0]-group[i][0]) # if i=0, then this will reach back around to the point at the end of the list, meaning group[i-1] will be group[-1] (Note: it's very important that the -1* component is different when looking back to when looking forward)
                n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
                n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
                candidates_side_two.append((group[i][0]+offset_scalar*n_unit[0], group[i][1]+offset_scalar*n_unit[1]))

                #getting point from normal vector between point at i and point at i+1
                n = (group[i+1][1]-group[i][1], -1*(group[i+1][0]-group[i][0])) #get normal vector by flipping x & y components of the vector between the point at i and the point at i+1 and negating one of those componenets, which component you negate changes which normal vector is used
                n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
                n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
                # only add the point from this normal vector if it's not too close to the point from the previous normal vector (determined by offset_point_distance_proportion and offset_scalar)
                if euclidean_distance(candidates_side_two[0], (group[i][0]+offset_scalar*n_unit[0], group[i][1]+offset_scalar*n_unit[1])) > offset_point_distance_proportion*offset_scalar:
                    candidates_side_two.append((group[i][0]+offset_scalar*n_unit[0], group[i][1]+offset_scalar*n_unit[1]))

                # deciding which candidate side we should use
                use_side_one = True
                for candidate in candidates_side_one:
                    # checking if either of the points in candidate_side_one fall inside the shape
                    if polygon.contains(s.Point(candidate)): # if the candidate point falls inside the shape, we use side two
                        use_side_one = False
                
                if use_side_one: #we use side one
                    for candidate in candidates_side_one:
                        offset_group.append(candidate)
                else: #we use side two
                    for candidate in candidates_side_two:
                        if not polygon.contains(s.Point(candidate)): #also check these points for collisions (it's possible to get all points from normal vectors at a fixed offset to result in a collision)
                            offset_group.append(candidate)
                        else: #if it turns out that these points are also bad, we just won't add them --> TODO: come up with system to try to find an okay point that works
                            pass

            # now we do the same for the last item, but wrap around to the first item when doing getting the second normal vector
            #consider one side of the boundary
            candidates_side_one = []
            n = (group[-2][1]-group[-1][1], -1*(group[-2][0]-group[-1][0])) # normal vector gotten from the vector from last point to second to last point 
            n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
            n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
            candidates_side_one.append((group[-1][0]+offset_scalar*n_unit[0], group[-1][1]+offset_scalar*n_unit[1]))

            n = (-1*(group[0][1]-group[-1][1]), group[0][0]-group[-1][0]) # normal vector gotten from the vector from the last point in the list to the first 
            n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
            n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
            if euclidean_distance(candidates_side_one[0], (group[-1][0]+offset_scalar*n_unit[0], group[-1][1]+offset_scalar*n_unit[1])) > offset_point_distance_proportion*offset_scalar:
                candidates_side_one.append((group[-1][0]+offset_scalar*n_unit[0], group[-1][1]+offset_scalar*n_unit[1]))

            #getting the possible boundary points on the other side of the boundary
            candidates_side_two = [] #contains two points on one side of the boundary curve
            #getting the point from the normal vector between point at i-1 and point at i
            n = (-1*(group[-2][1]-group[-1][1]), group[-2][0]-group[-1][0])
            n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
            n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
            candidates_side_two.append((group[-1][0]+offset_scalar*n_unit[0], group[-1][1]+offset_scalar*n_unit[1]))

            #getting point from normal vector between point at i and point at i+1
            n = (group[0][1]-group[-1][1], -1*(group[0][0]-group[-1][0])) #get normal vector by flipping x & y components of the vector between the point at i and the point at i+1 and negating one of those componenets, which component you negate changes which normal vector is used
            n_magnitude = math.sqrt(n[0]**2+n[1]**2) # magnitude of normal vector
            n_unit = (n[0]/n_magnitude, n[1]/n_magnitude) # unit normal vector 
            # only add the point from this normal vector if it's not too close to the point from the previous normal vector (determined by offset_point_distance_proportion and offset_scalar)
            if euclidean_distance(candidates_side_two[0], (group[-1][0]+offset_scalar*n_unit[0], group[-1][1]+offset_scalar*n_unit[1])) > offset_point_distance_proportion*offset_scalar:
                candidates_side_two.append((group[-1][0]+offset_scalar*n_unit[0], group[-1][1]+offset_scalar*n_unit[1]))

            # deciding which candidate side we should use
            use_side_one = True
            for candidate in candidates_side_one:
                # checking if either of the points in candidate_side_one fall inside the shape
                if polygon.contains(s.Point(candidate)): # if the candidate point falls inside the shape, we use side two
                    use_side_one = False
                    
            if use_side_one: #we use side one
                for candidate in candidates_side_one:
                    offset_group.append(candidate)
            else: #we use side two
                for candidate in candidates_side_two:
                    if not polygon.contains(s.Point(candidate)): #also check these points for collisions (it's possible to get all points from normal vectors at a fixed offset to result in a collision)
                        offset_group.append(candidate)
                    else: #if it turns out that these points are also bad, we just won't add them --> TODO: come up with system to try to find an okay point that works
                        pass

            offset_boundary_points.append(offset_group)

        # relocating outlier points on the offset boundary
        relocate_points(offset_boundary_points)

        # writing the offset boundary points to the obj file
        group_index = 0    
        for group in offset_boundary_points:
            #defining the vertices on the offset boundary in the obj file
            out.write(f"\n\n# Defining offset boundary points for group: {groupID_array[reduced_boundaries[group_index][0][1]][reduced_boundaries[group_index][0][0]]}")
            for offset_point in group:
                out.write(f"\nv {offset_point[0]} {offset_point[1]} {-1*extrusion}")
                obj_vertex_numbers[(offset_point[0], offset_point[1], -1*extrusion)] = vertex_index
                vertex_index += 1

                out.write(f"\nv {offset_point[0]} {offset_point[1]} 0")
                obj_vertex_numbers[(offset_point[0], offset_point[1], 0)] = vertex_index
                vertex_index += 1
            
            #making connecting faces between the offset boundary points and the extruded points
            out.write("\n#defining connecting faces")
            for i in range(len(group)-1):
                out.write(f"\nf {obj_vertex_numbers[(group[i][0], group[i][1], 0)]} {obj_vertex_numbers[(group[i+1][0], group[i+1][1], 0)]} {obj_vertex_numbers[(group[i+1][0], group[i+1][1], -1*extrusion)]} {obj_vertex_numbers[(group[i][0], group[i][1], -1*extrusion)]}\n")
            #do last vertex
            out.write(f"\nf {obj_vertex_numbers[(group[-1][0], group[-1][1], 0)]} {obj_vertex_numbers[(group[0][0], group[0][1], 0)]} {obj_vertex_numbers[(group[0][0], group[0][1], -1*extrusion)]} {obj_vertex_numbers[(group[-1][0], group[-1][1], -1*extrusion)]}\n")
                    

            #making a back face with the extruded offset boundary points
            face = "\n#making a back face for the offset\nf"
            for offset_point in group:
                face += f" {obj_vertex_numbers[(offset_point[0],offset_point[1],-1*extrusion)]}"
            out.write(face)

            group_index +=1
            

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


    # #saving the pixels in reduced boundaries, each pixel in the boundary's color is based on it's position in the list
    # output_pixels = [[(255,255,255) for x in range(x_size)] for y in range(y_size)]
    # for g in reduced_boundaries:
    #     color_value = 0
    #     for p in g:
    #         output_pixels[p[1]][p[0]] = (color_value,0,0)
    #         color_value += 5
    #         if color_value >= 255:
    #             color_value = 255

    # output = []
    # for y in output_pixels:
    #     for x in y:
    #         output.append(x)

    reference_image = Image.new(mode="RGB", size=reference_size)
    reference_image.putdata(output)
    reference_image.save(f"images\{reference_name}_marked_in_order_borders_reduced_by_{int(100*reduction_factor)}_percent.png")

save_progression_images()