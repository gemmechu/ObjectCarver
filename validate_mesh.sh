SCENE=scan_3
SCENE_TYPE=small_scene
python training/obj_separation/${SCENE_TYPE}/exp_runner.py --mode validate_mesh --conf ./confs/wmask_obj_separation.conf --case ${SCENE} --is_continue