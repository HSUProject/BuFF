@echo off
cd data/nerf
rmdir /s /q demo
mkdir demo
cd ../../../../../Desktop
move demo.mp4 ../Documents/GitHub/BuFF/data/nerf/demo
cd ../Documents/GitHub/BuFF/data/nerf/demo
python ../../../scripts/colmap2nerf.py --video_in demo.mp4 --video_fps 2 --run_colmap --aabb_scale 2
cd ../../../
instant-ngp.exe --scene=data/nerf/demo
pause