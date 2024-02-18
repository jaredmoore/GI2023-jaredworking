from PIL import Image, ImageDraw, ImageChops,  ImageOps
import opensimplex
from perlin_noise import PerlinNoise
from pixelsort import pixelsort
import random
import math
import numpy as np
from settings import *


### Utility functions
# map function similar to p5.js
def p5map(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


# constrain value to range
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


# https://stackoverflow.com/questions/3098406/root-mean-square-difference-between-two-images-using-python-and-pil
def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value * ((idx % 256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares / float(im1.size[0] * im1.size[1]))
    return rms


# c/o https://codereview.stackexchange.com/questions/55902/fastest-way-to-count-non-zero-pixels-using-python-and-pillow
def count_nonblack_pil(img):
    bbox = img.getbbox()
    if not bbox: return 0
    return sum(
        img.crop(bbox).point(lambda x: 255
                             if x else 0).convert("L").point(bool).getdata())


###


def pixelSort(img, params):
    return pixelsort(
        img,
        angle=int(params[0]),
        interval_function=params[1],
        sorting_function=params[2],
        randomness=float(params[3]),
        #        char_length=float(params[4]),
        lower_threshold=float(params[5]),
        upper_threshold=float(params[6]))
    """
        image: Image.Image,
        mask_image: typing.Optional[Image.Image] = None,
        interval_image: typing.Optional[Image.Image] = None,
        randomness: float = DEFAULTS["randomness"],
        char_length: float = DEFAULTS["char_length"],
        sorting_function: typing.Literal["lightness", "hue", "saturation", "intensity", "minimum"] = DEFAULTS[
            "sorting_function"],
        interval_function: typing.Literal["random", "threshold", "edges", "waves", "file", "file-edges", "none"] =
        DEFAULTS["interval_function"],
        lower_threshold: float = DEFAULTS["lower_threshold"],
        upper_threshold: float = DEFAULTS["upper_threshold"],
        angle: float = DEFAULTS["angle"]
    """


# "simple" PIL dithering
def simpleDither(img):
    dithered = img.convert(mode="1")
    dithered = dithered.convert("RGBA")
    return dithered


# "Simple" Drunkards walk
def drunkardsWalk(
        img,
        palette=None,
        pointSize=1,
        life=None,
        startX=None,
        startY=None,
        #   col=None,
        numSteps=None):
    # randomly set parameters if not specified
    if startX == None:
        startX = random.randint(0, DIM[0] - 1)

    if startY == None:
        startY = random.randint(0, DIM[1] - 1)

    if life == None:
        life = random.randint(int(DIM[0] * 0.5), int(5 * DIM[0]))

    if palette == None:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        a = random.randint(20, 255)
        col = (r, g, b, a)
    else:
        palette = getPaletteValues(palette)
        random.shuffle(palette)
        col = palette[0]

    if numSteps == None:
        numSteps = random.randint(5, 200)

    draw = ImageDraw.Draw(img)

    # directions [x, y]
    dirs = [
        [-1, -1],  # top left
        [0, -1],  # top 
        [1, -1],  # top right
        [-1, 0],  # middle left
        [0, 0],  # middle 
        [1, 0],  # middle right
        [-1, 1],  # bottom left
        [0, 1],  # bottom 
        [1, 1],  # bottom right
    ]

    l = 0
    x = startX
    y = startY
    while numSteps > 0:
        d = random.choice(dirs)
        while l < life:
            draw.rectangle([x, y, x + pointSize, y + pointSize], fill=col)

            # allow the drunkard to keep walking to 'branch' out a bit
            if (random.random() > 0.75):
                d = random.choice(dirs)
            x += d[0]
            y += d[1]

            x = constrain(x, 0, DIM[0])
            y = constrain(y, 0, DIM[1])
            l += 1

        numSteps -= 1
        l = 0

        if (random.random() > 0.85):
            startX = random.randint(0, DIM[0] - 1)
            startY = random.randint(0, DIM[1] - 1)

            # new color!
            random.shuffle(palette)
            col = palette[0]

        x = startX
        y = startY

    return


def stippledBG(img, fill, DIM):
    draw = ImageDraw.Draw(img)
    for y in range(DIM[1]):
        num = int(DIM[0] * p5map(y, 0, DIM[1], 0.01, 0.2))
        for _ in range(num):
            x = random.randint(0, DIM[0] - 1)
            draw.point((x, y), fill)


# Noise from opensimplex.noise returns [-1,1]
def flowField(img,
              cellsize,
              numrows,
              numcols,
              fill,
              flowType,
              multX=0.01,
              multY=0.01):
    # unpack the string
    multX = float(multX)
    multY = float(multY)

    draw = ImageDraw.Draw(img)
    grid = []
    for r in range(numrows):
        grid.append([])
        for c in range(numcols):
            n = opensimplex.noise2(x=c * multX, y=r * multY)

            if (flowType == "curves"):
                grid[r].append(p5map(n, -1.0, 1.0, 0.0, 2.0 * math.pi))
            else:
                grid[r].append(
                    math.ceil((p5map(n, 0.0, 1.0, 0.0, 2.0 * math.pi) *
                               (math.pi / 4.0)) / (math.pi / 4.0)))

    particles = []
    for _ in range(1000):
        p = {
            'x': random.randint(0, numcols - 1),
            'y': random.randint(0, numrows - 1),
            'life': random.randint(numcols / 2, numcols)
        }
        particles.append(p)

    while len(particles) > 0:
        #print(len(particles))
        for i in range(len(particles) - 1, -1, -1):
            p = particles[i]
            draw.point((p['x'], p['y']), fill)

            angle = grid[int(p['y'])][int(p['x'])]

            p['x'] += math.cos(angle)
            p['y'] += math.sin(angle)
            p['life'] -= 1
            # print(p)

            if (p['x'] < 0 or p['x'] > numcols - 1 or p['y'] < 0
                    or p['y'] > numrows - 1 or p['life'] <= 0):
                particles.pop(i)
    return


# Based on https://p5js.org/examples/simulate-wolfram-ca.html
def WolframCARules(a, b, c, ruleset):
    if a == 1 and b == 1 and c == 1: return ruleset[0]
    if a == 1 and b == 1 and c == 0: return ruleset[1]
    if a == 1 and b == 0 and c == 1: return ruleset[2]
    if a == 1 and b == 0 and c == 0: return ruleset[3]
    if a == 0 and b == 1 and c == 1: return ruleset[4]
    if a == 0 and b == 1 and c == 0: return ruleset[5]
    if a == 0 and b == 0 and c == 1: return ruleset[6]
    if a == 0 and b == 0 and c == 0: return ruleset[7]
    return 0


def WolframCAGenerate(cells, generation, ruleset):
    nextgen = [0 for _ in range(len(cells))]
    for i in range(1, len(cells) - 1):
        left = cells[i - 1]
        middle = cells[i]
        right = cells[i + 1]
        nextgen[i] = WolframCARules(left, middle, right, ruleset)
    #cells = nextgen
    generation += 1
    return nextgen, generation


    # get list of hex values
def getPaletteValues(p):
    palette = p.split(" ")
    for i, hex in enumerate(palette):
        palette[i] = "#" + hex
    return palette


def WolframCA(img, palette):
    # setup
    draw = ImageDraw.Draw(img)

    palette = getPaletteValues(palette)
    random.shuffle(palette)
    main_col = palette[0]

    width, height = img.size
    w = 10
    h = (height // w) + 1
    cells = []
    generation = 0

    num_cells = (width // w) + 1
    cells = [0 for _ in range(num_cells)]

    # random starting point
    # TBD param
    if random.random() > 0.5:
        cells[len(cells) // 2] = 1
    else:
        cells[random.randint(0, len(cells) - 1)] = 1

    # standard wolfram rules
    # TBD param
    if random.random() > 0.5:
        ruleset = [0, 1, 0, 1, 1, 0, 1, 0]
    else:
        # random rules
        ruleset = []
        for _ in range(8):
            ruleset.append(random.choice([0, 1]))

    # draw and iterate
    col = (220, 220, 220)
    while generation < h:
        for i in range(len(cells)):
            x = i * w
            y = generation * w
            if cells[i] == 1:
                col = main_col  #(220, 0, 220)
            else:
                col = (0, 0, 0)
            draw.rectangle([x, y, x + w, y + w], fill=col)

        cells, generation = WolframCAGenerate(cells, generation, ruleset)
    return


def flowField2(img, palette, flowtype, noisescale, resolution):
    draw = ImageDraw.Draw(img)

    # unpack strings
    noisescale = int(noisescale)
    resolution = int(resolution)

    # get list of hex values
    palette = getPaletteValues(palette)
    # palette = palette.split(" ")
    # for i, hex in enumerate(palette):
    #     palette[i] = "#" + hex

    particles = []
    noise = PerlinNoise()

    # add particles along top and bottom
    for x in range(0, DIM[0], resolution):
        r = random.random()
        if r < 0.5:
            p = {'x': x, 'y': 0, 'colour': random.choice(palette)}
            particles.append(p)
        else:
            p = {'x': x, 'y': DIM[1], 'colour': random.choice(palette)}
            particles.append(p)
        x += resolution

    # add particles along left and right sides
    for y in range(0, DIM[1], resolution):
        r = random.random()
        if r < 0.5:
            p = {'x': 0, 'y': y, 'colour': random.choice(palette)}
            particles.append(p)
        else:
            p = {'x': DIM[0], 'y': y, 'colour': random.choice(palette)}
            particles.append(p)
        y += resolution

    while len(particles) > 0:
        for i in range(len(particles) - 1, -1, -1):
            p = particles[i]

            draw.point((p['x'], p['y']), p['colour'])
            noiseval = noise([p['x'] / noisescale, p['y'] / noisescale])

            if (flowtype == "curvy"):
                angle = p5map(noiseval, -1.0, 1.0, 0.0, math.pi * 2.0)
            if (flowtype == "edgy"):
                angle = math.ceil(
                    p5map(noiseval, -1.0, 1.0, 0.0, math.pi * 2) *
                    (math.pi / 2)) / (math.pi / 2)

            p['x'] += math.cos(angle)
            p['y'] += math.sin(angle)

            # check edge
            if (p['x'] < 0 or p['x'] > DIM[0] or p['y'] < 0
                    or p['y'] > DIM[1]):
                particles.pop(i)
    return


def circlePacking(img, palette, limit):
    draw = ImageDraw.Draw(img)

    # unpack strings
    limit = int(limit)

    # get list of hex values
    palette = getPaletteValues(palette)
    circles = []
    total = 7  # circles to add each loop

    while True:
        count = 0
        failures = 0
        finished = False

        while count < total:
            # random centerpoint
            x = random.randrange(DIM[0])
            y = random.randrange(DIM[1])
            valid = True

            for c in circles:
                # distance between new circle centerpoint and existing circle centerpoint
                d = math.dist([x, y], [c['x'], c['y']])

                if d < c['radius'] + 3:
                    valid = False
                    break

            if valid:
                newC = {
                    'x': x,
                    'y': y,
                    'radius': 1,
                    'colour': random.choice(palette),
                    'growing': True
                }
                circles.append(newC)
                count += 1
            else:
                failures += 1
            if failures >= limit:
                finished = True
                break

        if finished:
            break

        # grow circles, check edges
        for c in circles:
            x = c['x']
            y = c['y']
            radius = c['radius']

            growing = c['growing']

            if growing:
                # check if circle hit canvas edge
                if x + radius >= DIM[0] or x - radius <= 0 or y + radius >= DIM[
                        1] or y - radius <= 0:
                    c['growing'] = False
                else:
                    # check if circle hit other circle
                    for c2 in circles:
                        x2 = c2['x']
                        y2 = c2['y']
                        radius2 = c2['radius']
                        if c != c2:
                            d = math.dist([x, y], [x2, y2])
                            # check if circles hit each other, with small buffer
                            if d - 4 < radius + radius2:
                                c['growing'] = False
                                break
            if growing:
                # grow
                c['radius'] += 1

    # display
    for c in circles:
        x = c['x']
        y = c['y']
        radius = c['radius']

        if radius == 1:
            pass
        else:
            draw.ellipse(xy=(x - radius, y - radius, x + radius, y + radius),
                         fill=c['colour'],
                         width=radius)

def invert(img):
    
    r,g,b,a = img.split()
    rgb_image = Image.merge('RGB', (r,g,b))

    inverted_image = ImageOps.invert(rgb_image)

    r2,g2,b2 = inverted_image.split()

    new_img = Image.merge('RGBA', (r2,g2,b2,a))
    
    mask = mask_random_rects(img.size, random.randint(4,8))
    img = Image.composite(new_img, img, mask)
    
    return img

def technique_mask(img, img2):
    # Take in two images and mask the second inside randomly generated mask.
    
    new_img = Image.new("RGBA", img.size, "white")
    flowField2(new_img, random.choice(palettes), 'curvy', random.randrange(200, 600), random.randrange(2, 5))
    
    mask = mask_random_rects(img.size)
    
    img = Image.composite(img2, img, mask)
    
    return img

def mask_random_rects(img_dim, num_rects=6):
    # Generate a mask of random rectangles and return.
    mask = Image.new("L", img_dim, 0)
    draw = ImageDraw.Draw(mask)
    # Create a random number and size of rectangles to add to the mask.
    for i in range(num_rects):
        start_x = int(random.uniform(0,0.8*img_dim[0]))
        start_y = int(random.uniform(0,0.8*img_dim[1]))
        end_x = random.randint(start_x+10,img_dim[0])
        end_y = random.randint(start_y+10,img_dim[1])
        draw.rectangle((start_x, start_y, end_x, end_y), fill=255)
        
    return mask

# ---

# Dithering c/o
## https://www.codementor.io/@isaib.cicourel/image-manipulation-in-python-du1089j1u

# Create a new image with the given size
def create_image(i, j):
    image = Image.new("RGBA", (i, j), BACKGROUND)
    return image


# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
        return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel


# Create a Grayscale version of the image
def convert_grayscale(image):
    # Get size
    width, height = image.size

    # Create new Image and a Pixel Map
    new = create_image(width, height)
    pixels = new.load()

    # Transform to grayscale
    for i in range(width):
        for j in range(height):
            # Get Pixel
            pixel = get_pixel(image, i, j)

            # Get R, G, B values (This are int from 0 to 255)
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]

            # Transform to grayscale
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

            # Set Pixel in new image
            pixels[i, j] = (int(gray), int(gray), int(gray))

    # Return new image
    return new


# Create a Half-tone version of the image
def convert_halftoning(image):
    # Get size
    width, height = image.size

    # Create new Image and a Pixel Map
    new = create_image(width, height)
    pixels = new.load()

    # Transform to half tones
    for i in range(0, width, 2):
        for j in range(0, height, 2):
            # Get Pixels
            p1 = get_pixel(image, i, j)
            p2 = get_pixel(image, i, j + 1)
            p3 = get_pixel(image, i + 1, j)
            p4 = get_pixel(image, i + 1, j + 1)

            # Transform to grayscale
            gray1 = (p1[0] * 0.299) + (p1[1] * 0.587) + (p1[2] * 0.114)
            gray2 = (p2[0] * 0.299) + (p2[1] * 0.587) + (p2[2] * 0.114)
            gray3 = (p3[0] * 0.299) + (p3[1] * 0.587) + (p3[2] * 0.114)
            gray4 = (p4[0] * 0.299) + (p4[1] * 0.587) + (p4[2] * 0.114)

            # Saturation Percentage
            sat = (gray1 + gray2 + gray3 + gray4) / 4

            # Draw white/black depending on saturation
            if sat > 223:
                pixels[i, j] = (255, 255, 255)  # White
                pixels[i, j + 1] = (255, 255, 255)  # White
                pixels[i + 1, j] = (255, 255, 255)  # White
                pixels[i + 1, j + 1] = (255, 255, 255)  # White
            elif sat > 159:
                pixels[i, j] = (255, 255, 255)  # White
                pixels[i, j + 1] = (0, 0, 0)  # Black
                pixels[i + 1, j] = (255, 255, 255)  # White
                pixels[i + 1, j + 1] = (255, 255, 255)  # White
            elif sat > 95:
                pixels[i, j] = (255, 255, 255)  # White
                pixels[i, j + 1] = (0, 0, 0)  # Black
                pixels[i + 1, j] = (0, 0, 0)  # Black
                pixels[i + 1, j + 1] = (255, 255, 255)  # White
            elif sat > 32:
                pixels[i, j] = (0, 0, 0)  # Black
                pixels[i, j + 1] = (255, 255, 255)  # White
                pixels[i + 1, j] = (0, 0, 0)  # Black
                pixels[i + 1, j + 1] = (0, 0, 0)  # Black
            else:
                pixels[i, j] = (0, 0, 0)  # Black
                pixels[i, j + 1] = (0, 0, 0)  # Black
                pixels[i + 1, j] = (0, 0, 0)  # Black
                pixels[i + 1, j + 1] = (0, 0, 0)  # Black

    # Return new image
    return new


# Return color value depending on quadrant and saturation
def get_saturation(value, quadrant):
    if value > 223:
        return 255
    elif value > 159:
        if quadrant != 1:
            return 255

        return 0
    elif value > 95:
        if quadrant == 0 or quadrant == 3:
            return 255

        return 0
    elif value > 32:
        if quadrant == 1:
            return 255

        return 0
    else:
        return 0


# Create a dithered version of the image
def convert_dithering(image):
    # Get size
    width, height = image.size

    # Create new Image and a Pixel Map
    new = create_image(width, height)
    pixels = new.load()

    # Transform to half tones
    for i in range(0, width, 2):
        for j in range(0, height, 2):
            # Get Pixels
            p1 = get_pixel(image, i, j)
            p2 = get_pixel(image, i, j + 1)
            p3 = get_pixel(image, i + 1, j)
            p4 = get_pixel(image, i + 1, j + 1)

            # Color Saturation by RGB channel
            red = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
            green = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
            blue = (p1[2] + p2[2] + p3[2] + p4[2]) / 4

            # Results by channel
            r = [0, 0, 0, 0]
            g = [0, 0, 0, 0]
            b = [0, 0, 0, 0]

            # Get Quadrant Color
            for x in range(0, 4):
                r[x] = get_saturation(red, x)
                g[x] = get_saturation(green, x)
                b[x] = get_saturation(blue, x)

            # Set Dithered Colors
            pixels[i, j] = (r[0], g[0], b[0])
            pixels[i, j + 1] = (r[1], g[1], b[1])
            pixels[i + 1, j] = (r[2], g[2], b[2])
            pixels[i + 1, j + 1] = (r[3], g[3], b[3])

    # Return new image
    return new


# Create a Primary Colors version of the image
def convert_primary(image):
    # Get size
    width, height = image.size

    # Create new Image and a Pixel Map
    new = create_image(width, height)
    pixels = new.load()

    # Transform to primary
    for i in range(width):
        for j in range(height):
            # Get Pixel
            pixel = get_pixel(image, i, j)

            # Get R, G, B values (This are int from 0 to 255)
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]

            # Transform to primary
            if red > 127:
                red = 255
            else:
                red = 0
            if green > 127:
                green = 255
            else:
                green = 0
            if blue > 127:
                blue = 255
            else:
                blue = 0

            # Set Pixel in new image
            pixels[i, j] = (int(red), int(green), int(blue))

    # Return new image
    return new


def lineContours(img, palette, num_lines=80, line_length=20, angle_incr=10, x_incr=2, y_incr=2):
    draw = ImageDraw.Draw(img)

    # get list of hex values
    palette = getPaletteValues(palette)
    
    width, height = img.size
    
    x = random.randrange(width)
    y = random.randrange(height)
    
    angle = random.randrange(180)
    
    direction_x = random.choice([-1, 1])
    direction_y = random.choice([-1, 1])
    direction_angle = random.choice([-1, 1])
    
    for i in range(num_lines):
        # Calculate the (x,y) coordinates of the end point of the line to draw based on the angle.
        
        x1 = x + line_length * math.cos(math.radians(angle))
        x2 = x - line_length * math.cos(math.radians(angle))
        
        y1 = y + line_length * math.sin(math.radians(angle))
        y2 = y - line_length * math.sin(math.radians(angle))
        
        draw.line((x1,y1,x2,y2),  fill=(255, 255, 255, 255), width=2)
        
        x = x + (random.randrange(x_incr) * direction_x)
        y = y + (random.randrange(y_incr) * direction_y)
        angle = angle + (random.randrange(angle_incr) * direction_angle)
        
        if x > width or x < 0:
            direction_x *= -1
            
        if y > height or y < 0:
            direction_y *= -1
            
        if angle > 180 or angle < 0:
            direction_angle *= -1
            
def gravityParticle(img, palette, num_steps=1000, radius=2, step_update=False):
    draw = ImageDraw.Draw(img)

    # get list of hex values
    palette = getPaletteValues(palette)
    
    width, height = img.size
    
    angle = random.randrange(360)
    length = width/random.randrange(1,10)
    cur_pos = 0
    
    x = width//4 + random.randrange(width)//2
    y = height//4 + random.randrange(height)//2
    
    angle_update = random.randrange(1,5)
    position_update = random.randrange(1,10)
    
    rotate_direction = random.choice([-1, 1])
    direction = random.choice([-1, 1])
    
    for i in range(num_steps):
        if step_update:
            angle_update = random.randrange(1,5)
            position_update = random.randrange(4,20)
        
        # Calculate the (x,y) coordinates of the end point of the line to draw based on the angle.
        x1 = x + random.randrange(-10,10) + cur_pos * math.cos(math.radians(angle))
        y1 = y + random.randrange(-10,10) + cur_pos * math.sin(math.radians(angle))
        
        draw.ellipse(xy=(x1 - radius, y1 - radius, x1 + radius, y1 + radius),
                         fill=(255,255,255,255),
                         width=radius)
        
        
        
        cur_pos = cur_pos + (position_update * direction)
        
        if cur_pos > length:
            direction *= -1
            cur_pos = length
        elif cur_pos < 0:
            direction *= -1
            cur_pos = 0
            
        angle = (angle + (angle_update * rotate_direction)) % 360 
        
def abstractRectangles(img, palette, num_rects=10):
    draw = ImageDraw.Draw(img)

    # get list of hex values
    palette = getPaletteValues(palette)
    
    width, height = img.size
    
    for _ in range(num_rects):
        x = width//20 + random.randrange(width-width//20)
        y = height//20 + random.randrange(height-height//20)
        
        rect_length = (width-x)//random.randrange(1,10)
        rect_height = (height-y)//random.randrange(1,10)
        
        rect_length = 1 if rect_length < 1 else rect_length
        rect_height = 1 if rect_height < 1 else rect_height
    
        mondrian_rectangle(img, palette, x, y, rect_length, rect_height, random.randint(0,20), random.randrange(0,400))
        
def mondrian_rectangle(img, palette, x, y, rect_len, rect_ht, rect_fill, line_overdraw):
    """ Draw a mondrian style rectangle with black lines exceeding the edges.

    Args:
        img (PIL image object): the image to draw on.
        palette (list): list of colors to choose from.
        x (int): center x location of the rectangle.
        y (int): center y location of the rectangle.
        rect_len (int): length of the rectangle.
        rect_ht (int): height of the rectangle.
        rect_fill (int): index to use to fill the rectangle with in (rgb) from palette (modulo to prevent error).
        line_overdraw (int): The amount of overdraw for the black lines.
    """
    draw = ImageDraw.Draw(img)
    draw.rectangle(xy=(x, y, x + rect_len, y + rect_ht),
                         fill=palette[rect_fill%len(palette)])
    draw.line((x-line_overdraw, y,                x+rect_len+line_overdraw,    y), fill="black", width=4)
    draw.line((x-line_overdraw, y+rect_ht,    x+rect_len+line_overdraw,    y+rect_ht), fill="black", width=4)
    draw.line((x,               y-line_overdraw,  x,                y+rect_ht+line_overdraw), fill="black", width=4)
    draw.line((x+rect_len,   y-line_overdraw,  x+rect_len,    y+rect_ht+line_overdraw), fill="black", width=4)
        
def staticShifter(img):
    # Create a band of static in an image by shifting ranges of pixels.
    width, height = img.size
    
    shift_pixel_range = random.randrange(height//8,height//2)
    
    start_shift = random.randrange(0, height-shift_pixel_range)
    
    shift_stride = random.randrange(1,5)
    
    for y in range(start_shift, start_shift+shift_pixel_range, shift_stride):
        line_shift = random.choice([-1,1]) * random.randrange(width//50, width//10)
        for y_shift in range(y, y+shift_stride):
            for x in range(width):
                pixel = img.getpixel((x,y_shift))
                img.putpixel(((x+line_shift)%width,y_shift), pixel)
                
def clifford_attractor(x, y, a=-1.4, b=1.7, c=1.0, d=0.7):
	'''Returns the change in arguments x and y according to 
	the Clifford map equation. Kwargs a, b, c, and d are specified
	as constants.
 
    Source: https://blbadger.github.io/clifford-attractor.html
	'''
	x_next = np.sin(a*y) + c*np.cos(a*x) 
	y_next = np.sin(b*x) + d*np.cos(b*y)
	return x_next, y_next

def clifford_sequence(img, x, y, color=(255,255,255,255), cliff_params=False):
    """ Create a clifford attractor sequence starting at the provided x and y."""
    # Parameters for num_iters and sine wave.
    num_iters = 400000
    a_phase_offset = np.radians(random.randrange(0, 360))
    b_phase_offset = np.radians(random.randrange(0, 360))
    
    draw = ImageDraw.Draw(img)
    
    if cliff_params:
        a, b, c, d = cliff_params
    else:
        a = random.random()*4 - 2
        b = random.random()*4 - 2
        c = random.random()*2
        d = random.random()*2
    
    print(f"[{x},{y},{a},{b},{c},{d}],")
    for i in range(num_iters):
        x, y = clifford_attractor(x, y, a, b, c, d)
        
        x_p = img.width//2 + x/np.pi*(img.width//2)
        y_p = img.height//2 + y/np.pi*(img.height//2)
        # draw.rectangle(xy=(x_p-1, y_p-1, x_p + 1, y_p + 1),
                        #  fill="black")
        draw.point((x_p, y_p), fill=color)
        
        # Could we use a third parameter for point size?
        # radius = random.randint(1,5)/5
        # draw.ellipse(xy=(x_p - radius, y_p - radius, x_p + radius, y_p + radius), fill=color)
        
        #a += random.random()*0.01 - 0.005
        #b += random.random()*0.01 - 0.005
        
        # a = ((np.sin(2*np.pi*0.000001*i + a_phase_offset) + 1)/2)*img.width
        #b = ((np.sin(2*np.pi*0.000001*i + b_phase_offset) + 1)/2)*img.height
        
        # if a < 0 or a > img.width or b < 0 or b > img.height:
        #     print(a,b)
        # draw.point((a,b), fill="green")
        
        
        
def cliffords(img, x, y, num_cliffs = 2):
    c_params = [
        [-1.314116732645811,-1.4488445790982132,0.48350173248983297,1.0231018578214124],
        [1.5241206408018835,-0.31494437751003845,1.5478918011471245,0.25993191444300034]
    ]
    colors = ((0,0,0,255),(50,50,150,255))
    for n in range(num_cliffs):
        clifford_sequence(img, x-400*n, y+400*n, color=colors[n])#, cliff_params=c_params[n])
        
def nested_cliffords(img, x, y):
    colors = ((0,0,0,255),(50,50,150,255))
    
    num_iters = 100000
    draw = ImageDraw.Draw(img)
    
    a = random.random()*4 - 2
    b = random.random()*2
    c = random.random()*2
    d = random.random()*2
    
    print(f"{a},{b},{c},{d}")
    for i in range(num_iters):
        x, y = clifford_attractor(x, y, a, b, c, d)
        x, y = clifford_attractor(x, y, x, y, c, d)
        
        x_p = img.width//2 + x/np.pi*(img.width//2)
        y_p = img.height//2 + y/np.pi*(img.height//2)
        # draw.rectangle(xy=(x_p-1, y_p-1, x_p + 1, y_p + 1),
                        #  fill="black")
        draw.point((x_p, y_p), fill="black")        
    