#!/bin/bash


python3 ../bricklayers.py ../sample_gcode/Sample_BrickLayersChallengeSimple_5walls.gcode \
	-outputFile ../sample_brick/SampleBrickLayersChallengeSimple_5walls_brick.gcode \
	-extrusionMultiplier 1.05 \
	-verbosity 1


python3 ../bricklayers.py ../sample_gcode/Sample_3DBenchy_5walls_classic.gcode \
	-outputFolder ../sample_brick/ \
    -outputFilePostfix _brick \
	-extrusionMultiplier 1.05 \
	-startAtLayer 3 \
	-ignoreLayers \
	-ignoreLayersFromTo \
	-enabled 1 \
	-verbosity 2
