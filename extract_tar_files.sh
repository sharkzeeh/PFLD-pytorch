#!/bin/sh
tar -xzf data/WFLW/WFLW_annotations.tar.gz -C data/WFLW
tar -xzf data/WFLW/WFLW_images.tar.gz -C data/WFLW

# moving Mirror98.txt
mv data/Mirror98.txt data/WFLW/WFLW_annotations/Mirror98.txt