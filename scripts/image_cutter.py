import json
import os
import shutil
from PIL import Image
from tqdm import tqdm
import random

WALDO_IMAGES_PATH = 'Where_is_Waldo_AI_solver/original-images/'
WALDO_IMAGES_FMT = '.jpg'
WALDO_IMAGES_POSITIONS = 'Where_is_Waldo_AI_solver/scripts/waldo_positions.json'
OUTPUT_DIR = 'output-DS-images'
OUTPUT_DIR_TEST = OUTPUT_DIR + '/test'
OUTPUT_DIR_TRAIN = OUTPUT_DIR + '/train'
BLOCK_SIZE = 64
NB_GRIDS = 16

TEST_IMAGES = [14, 15, 17, 18, 19]


class WaldoImage:
    """A class that stores a waldo image with its relevant characteristics"""

    REQUIRED_PERCENTAGE_WALDO = 75
    # The following is a probability, not a percentage
    PROBABILITY_KEEP_NOT_WALDO = 0.0025

    def __init__(self, img_id:int, waldo_corner_1:tuple, waldo_corner_2:tuple):
        image_path = WALDO_IMAGES_PATH+str(img_id)+WALDO_IMAGES_FMT
        self.img_id = img_id
        self.img = Image.open(image_path)
        self.waldo_corner_1 = waldo_corner_1
        self.waldo_corner_2 = waldo_corner_2

    def is_waldo_here(self, left, right, top, bottom):
        # Compute area of union
        union_area = max(0, min(self.waldo_corner_2[0], right) - max(self.waldo_corner_1[0], left)) * max(0, min(self.waldo_corner_2[1], bottom) - max(self.waldo_corner_1[1], top))
        # Compute area of waldo
        waldo_area = (abs(self.waldo_corner_2[0]-self.waldo_corner_1[0])) * (abs(self.waldo_corner_2[1]-self.waldo_corner_1[1]))

        #return self.waldo_corner_1[0]>=left and self.waldo_corner_1[1]>=top and self.waldo_corner_2[0]<=right and self.waldo_corner_2[1]<=bottom
        # Compute percentage of waldo in the input bounding box
        return 100*float(union_area)/(waldo_area) >= WaldoImage.REQUIRED_PERCENTAGE_WALDO

    def crop(self, output_dir:str, block_size:int, grid_offset_divider:int):
        """Creates labeled sub-images of size block_size in directory output_dir.
        A cropping grid is created according to the size block_size.
        Grid is then moved by block_size/grid_offset_divider in x direction, y direction, and both directions.
        The operation is performed grid_offset_divider times.
        """
        # Recover size of image
        width, height = self.img.size

        # For all possible cropping grids
        for i_grid in range(grid_offset_divider):
            x_offset = int(block_size*i_grid/grid_offset_divider)
            nb_blocks_x = int((width-x_offset)/block_size)
            for j_grid in range(grid_offset_divider):
                y_offset = int(block_size*j_grid/grid_offset_divider)
                nb_blocks_y = int((height-y_offset)/block_size)

                # Crop all images in this grid
                for i in range(nb_blocks_x):
                    for j in range(nb_blocks_y):
                        # Coordinates of current block
                        left = x_offset + i*block_size
                        top = y_offset + j*block_size
                        right = left + block_size
                        bottom = top + block_size
                        # Is waldo in the current block ?
                        here_he_is = self.is_waldo_here(left, right, top, bottom)

                        # Save only a small amount of not waldo images
                        if (not here_he_is and random.random() < WaldoImage.PROBABILITY_KEEP_NOT_WALDO) or here_he_is:
                            # Create a decent file name
                            filename = ('waldo' if here_he_is else 'notwaldo') + '-' + str(self.img_id) + '_' + str(left) + '_' + str(top) + '.png'
                            # Crop and save block
                            cropped = self.img.crop((left, top, right, bottom))
                            cropped.save(output_dir+'/'+filename)
                

def main():
    # Create a list of Waldo images objects
    waldo_images = []
    # Open the json file containing waldo positions
    with open(WALDO_IMAGES_POSITIONS) as json_file:
        data = json.load(json_file)
        # Create a WaldoImage object for each entry in the json file
        for pos in data:
            waldo_images.append(WaldoImage(pos['image'], pos['corner_1'], pos['corner_2']))

    # Clean output directory
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.mkdir(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR_TRAIN)
    os.mkdir(OUTPUT_DIR_TEST)
    # Process each waldo image
    print("Warning, the following process may be quite long")
    for waldo_image in tqdm(waldo_images):
        if waldo_image.img_id in TEST_IMAGES:
          output_dir = OUTPUT_DIR_TEST+'/'+str(waldo_image.img_id)
        else:
          output_dir = OUTPUT_DIR_TRAIN+'/'+str(waldo_image.img_id)
        os.mkdir(output_dir)
        waldo_image.crop(output_dir, BLOCK_SIZE, NB_GRIDS)


if __name__ == '__main__':
    main()
