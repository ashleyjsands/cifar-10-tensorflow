echo "Start host_notebooks.sh" >> /tmp/my-log
source `which virtualenvwrapper.sh`
workon tensorflow_master
cd ~/code/cifar-10-tensorflow/
echo "host_notebooks.sh: about to launch ipython" >> /tmp/my-log
ipython notebook --ip="*" --certfile=mycert.pem
