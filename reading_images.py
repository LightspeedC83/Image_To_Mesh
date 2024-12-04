from PIL import Image

#opening the reference image
reference_name = "test"
reference_path = f"{reference_name}.jpg"

reference_image = Image.open(reference_path)
reference_pixels = list(reference_image.getdata())
reference_size = reference_image.size # getting the refence image's size in (x-pixles,y-pixels) format


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


# creating pixel objects that can also hold other data (we'll hold them in a 2d list)
class Pixel(): #defining a pixel class
    def __init__(self, color, id, coordinates):
        self.color = color
        self.id = id
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.groupID = None
        self.isBorder = False
    
    def set_groupID(self, groupID):
        self.groupID = groupID

    def __str__(self):
        return(f"{self.color} | {self.id} | {self.coordinates}")

pixel_objects = [] # a 2D list that will hold objects of the pixel class, access a pixel via pixel_objects[y][x]
temp_list = []
i=0
for p in bw_reference: #converting all pixel values in bw_reference to Pixel objects and storing them in pixel_objects accordingly
    if len(temp_list) == reference_size[0]:
        pixel_objects.append(temp_list)
        temp_list = []
    
    temp_list.append(Pixel(p,i, (i%reference_size[0], len(pixel_objects))))

    i+=1
pixel_objects.append(temp_list)


#defining terminal testing functions
def print_by_group():
    for x in pixel_objects:
        temp = ""
        for p in x:
            if p.groupID == None:
                temp += "0 "
            else:
                temp += str(p.groupID+1) + " "
        print(temp)

def print_by_value():
    for x in pixel_objects:
        temp = ""
        for p in x:
            if p.color == (255,255,255):
                temp += ". "
            else:
                temp += "# "
        print(temp)


# now we do a flood fill to assign groupIDs to all pixel objects (white pixels have groupID None)
visited_spots = [[0 for x in range(reference_size[0])]for y in range(reference_size[1])] # a key of visited spots, 0 means we haven't visited, 1 means we have. use this for any given pixel to see if we've visited
points_by_group = []


group_index = 0
for y in range(reference_size[1]-1):
    for x in range(reference_size[0]-1):
        curr = pixel_objects[y][x]

        if visited_spots[y][x] == 0: # if the pixel under consideration hasn't been visited

            if curr.color == (255,255,255): #if pixel is white we mark as visited and move on
                visited_spots[y][x] = 1 
            else:
                queue = [] #create a queue
                queue.append(curr) 
                group = []
                while len(queue)!=0: #while queue isn't empty we keep looking at neighbors of the queue and assigning them IDs of said group
                    upNext = queue.pop(0)
                    upNext.groupID = group_index #assigning the group index for this group
                    group.append(upNext)
                    visited_spots[upNext.y][upNext.x] = 1 #marking this pixel as visited
                    for next_point in [[upNext.x-1, upNext.y], [upNext.x+1, upNext.y], [upNext.x, upNext.y-1], [upNext.x, upNext.y+1]]:
                        try:                          
                            if visited_spots[next_point[1]][next_point[0]] == 0 and pixel_objects[next_point[1]][next_point[0]].color != (255,255,255): #if the pixel at the potential neighbor position isn't already visited and isn't white, we add to queue to check
                                queue.append(pixel_objects[next_point[1]][next_point[0]])  
                        except: #if there is no pixel at this potential position
                            pass
                group_index +=1
                points_by_group.append(group)


# going through and marking the boundary pixels for each pixel group
for x in pixel_objects:
    for curr in x:
        for next_point in [[curr.x-1, curr.y], [curr.x+1, curr.y], [curr.x, curr.y-1], [curr.x, curr.y+1]]: 
            
            try:                          
                if pixel_objects[next_point[1]][next_point[0]].color == (255,255,255) and curr.groupID != None: #if the pixel at the potential neighbor position is white, we mark curr as a border pixel (curr also has to be part of a group)
                    curr.isBorder = True
            except:
                pass


output_pixels = []
for x in pixel_objects:
    for p in x:
        if p.isBorder:
            output_pixels.append((255,0,0))
        else:
            output_pixels.append((255,255,255))

reference_image = Image.new(mode="RGB", size=reference_size)
reference_image.putdata(output_pixels)
reference_image.save(f"{reference_name}_borders_marked.jpg")