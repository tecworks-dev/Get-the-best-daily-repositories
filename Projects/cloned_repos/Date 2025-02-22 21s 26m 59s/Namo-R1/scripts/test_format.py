from namo.utils.process_utils import convert_image_tags


# a = convert_image_tags('what in these images?\n<image>')
a = convert_image_tags("what in these images?\n<image>\n<image>")
print(a)
