models="./models/A_temp${1} ./models/A_temp${1}_distill"
python fgm_transfer.py --sources ${models} --targets ${models} --rand false
