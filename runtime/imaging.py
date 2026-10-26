import math
from PIL import Image


def transform(image, config):
    """Apply the crop tool's ordered pixel-space transforms."""
    assert list(image.size) == config['source_size'], image.size
    for op in config['operations']:
        if op['type'] == 'rotate':
            angle = math.radians(op['clockwise_degrees'])
            c, s = math.cos(angle), math.sin(angle)
            w, h = image.size
            width = math.ceil(abs(w*c) + abs(h*s))
            height = math.ceil(abs(w*s) + abs(h*c))
            matrix = (c, s, w/2-c*width/2-s*height/2,
                      -s, c, h/2+s*width/2-c*height/2)
            image = image.transform((width, height), Image.Transform.AFFINE,
                                    matrix, resample=Image.Resampling.BICUBIC,
                                    fillcolor=op['fill'])
        elif op['type'] == 'crop':
            x, y, w, h = (op[k] for k in ('x', 'y', 'width', 'height'))
            image = image.crop((x, y, x+w, y+h))
        else:
            raise ValueError(op['type'])
    assert list(image.size) == config['output_size'], image.size
    return image
