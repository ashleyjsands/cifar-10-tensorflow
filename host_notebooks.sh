#echo "Start host_notebooks.sh" >> /tmp/my-log
source `which virtualenvwrapper.sh`
workon tensorflow_py3_0.10
cd ~/code/cifar-10-tensorflow/
#echo "host_notebooks.sh: about to launch ipython" >> /tmp/my-log
ipython notebook --ip="*" --certfile=mycert.pem
