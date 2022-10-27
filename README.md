# slam
slam algorithm


# how to
```
# generate points
python3 main.py

node server.js
# and open display/js/site.html in browser

# linux only
./display/build/src/display - display points (wasd - move, er - next,previous frame, leftmousebutton & mosue - move camera)
```
# build display (linux)
```
cd display

mkdir build

cmake .. .

make

./src/display
```
# libs
opengl

sfml

opencv-python

opencv-contrib-python

kdtree
