# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
git clone https://github.com/AMD-AGI/GEAK-agent geak-openevolve 
cd geak-openevolve
git checkout geak-openevolve
pip install -e .

git clone https://github.com/AMD-AGI/GEAK-eval.git GEAK-eval-OE
cd GEAK-eval-OE 
git checkout geak-oe 
pip install -e . --no-deps 
cd ..