# Image To Mesh
## Goal & summary
The end goal of this project is to take in an image and create a mesh from it that can be used for 3D modeling or 3D printing.
## Image Processing
The program starts by taking in an image, it converts the image to black and white (by which I do not mean greyscale, but that every pixel is either black or white) based on a value if the average value of the RGB values of the pixel falls below, the pixel is considered black. We only care about the black pixels. We then do a flood fill algorithm on the image to identify a group ID number for each black pixel (white pixels have a group ID number of None). Then we find the border pixels of each group. This processing gives us the following 2D arrays in which the data for a particular pixel at coordiantes (x,y) is held:
- pixel_array - 2D array pixel_array[y][x] yeilds True if there is a black pixel at (x,y), False if not
- groupID_array - 2D array pixel_array[y][x] yeilds yeilds the group ID of a pixel at (x,y), None if there is not black pixel there
- boundary_pixels - 2D array boundary_pixels[y][x] yeilds True if there (x,y) has a boundary pixel, False if not
- points_by_group - a python list that holds groups, each group is a list of coordinate values

All of this processing is done with numpy arrays, which make the program much faster than if we used python's lists

