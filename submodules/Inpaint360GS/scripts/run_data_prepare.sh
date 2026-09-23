#!/bin/bash

scene="doppelherz"
python convert.py -s data/inpaint360/${scene}/train_and_test/ --resize --magick_executable convert
python tools/separate_train_test_ply.py --scene ${scene}