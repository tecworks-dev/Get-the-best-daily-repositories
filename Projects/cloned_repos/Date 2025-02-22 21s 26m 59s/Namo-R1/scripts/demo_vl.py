from namo.api.vl import VLInfer


vl = VLInfer(model_type="qwen2.5-vl")
# vl.generate("what is funny in this image?", "images/extreme_ironing.jpg")
# vl.generate("Outline the position of each car, output in json format", "images/extreme_ironing.jpg")
# vl.generate("Locate the person ironing cloth", "images/extreme_ironing.jpg")
# vl.generate("Point the blue shirt", "images/extreme_ironing.jpg")
vl.generate("Point all the cars in image", "images/extreme_ironing.jpg")
