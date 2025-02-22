from .image_processing_namo import NamoImageProcessor
from transformers import AutoImageProcessor

AutoImageProcessor.register(
    NamoImageProcessor, slow_image_processor_class=NamoImageProcessor
)
