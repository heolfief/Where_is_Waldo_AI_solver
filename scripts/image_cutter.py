import json
import os
import shutil
from PIL import Image

WALDO_IMAGES_PATH = './original-images/'
WALDO_IMAGES_FMT = '.jpg'
WALDO_IMAGES_POSITIONS = 'waldo_positions.json'
OUTPUT_DIR = 'output-images'
BLOCK_SIZE = 128
NB_GRIDS = 4

class WaldoImage:
    """A class that stores a waldo image with its relevant characteristics"""

    def __init__(self, img_id:int, waldo_corner_1:tuple, waldo_corner_2:tuple):
        image_path = WALDO_IMAGES_PATH+str(img_id)+WALDO_IMAGES_FMT
        self.img_id = img_id
        self.img = Image.open(image_path)
        self.waldo_corner_1 = waldo_corner_1
        self.waldo_corner_2 = waldo_corner_2

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
                        here_he_is = self.waldo_corner_1[0]>=left and self.waldo_corner_1[1]>=top and self.waldo_corner_2[0]<=right and self.waldo_corner_2[1]<=bottom
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
    # Process each waldo image
    for waldo_image in waldo_images:
        output_dir = OUTPUT_DIR+'/'+str(waldo_image.img_id)
        os.mkdir(output_dir)
        waldo_image.crop(output_dir, BLOCK_SIZE, NB_GRIDS)


if __name__ == '__main__':
    main()
