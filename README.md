# Computer_vision
computervision, image processing and machine learning


## Finding Contours
python3 hello.py


## Reading Threshold, Gray, Finding objects

python3 image_analysis.py -i testris.jpeg

## Object detection - animals
1.Download the appropriate release here: https://github.com/google/protobuf/releases

2.Unzip the folder

3.Enter the folder and run ./autogen.sh && ./configure && make

4.If you run into this error: autoreconf: failed to run aclocal: No such file or directory, run brew install autoconf && brew install automake. And run the command from step 3 again.

5.Then run these other commands. They should run without issues
```
$ make check
$ sudo make install
$ which protoc
$ protoc --version
```
6.cd models/research/

7.protoc object_detection/protos/*.proto --python_out=.

8.cp object_detection/packages/tf2/setup.py .

9.python -m pip install .
