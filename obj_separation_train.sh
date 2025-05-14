
#--------------- For small scene ---------------#
SCENE=scan_3 
CONF_PATH=./confs/wmask_obj_separation.conf 
python training/obj_separation/small_scene/exp_runner.py --mode train --conf ${CONF_PATH} --case ${SCENE} --is_continue 

#--------------- For indoor scene ---------------#
# SCENE=scannet_scan1 
# CONF_PATH=./confs/scannet.conf
# python training/obj_separation/indoor/exp_runner.py --mode train --conf ${CONF_PATH} --exp_name ${SCENE}b --scene_name ${SCENE} --is_continue

