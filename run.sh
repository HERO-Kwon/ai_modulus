
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
           --runtime nvidia -v $(pwd):/ai_modulus \
           -p 8888:8888 -p 6006:6006 -it --rm modulus:22.09 bash
