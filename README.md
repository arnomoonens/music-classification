# Music classification
Classification of MIDI music as a project for the course [Current Trends in Artificial Intelligence](http://ai.vub.ac.be/courses/2015-2016/current-trends-artificial-intelligence) at the [VUB](http://www.vub.ac.be/). Our team name is **Swinging Penguins**.

## Representation
As a representation, we will use n-grams. More information about this representation and why we will use it can be found [here](https://github.com/arnomoonens/music-classification/blob/master/SwingingPenguins-Phase1.pdf).

## Running our code
First, some Python libraries need to be installed. This can be done by executing `pip install -r requirements.txt`.
Then, you need to add a file named `.theanorc` to your home folder. the contents of these are:
```
[gcc]
cxxflags=-march=ivybridge
[global]
floatX=float32
```
It is possible that you have to change `ivybridge` to match your cpu architecture type.
Afterwards, you can run our code by executing the provided Perl script (`crossvalidate.pl`), providing the repo folder as argument. So basically: `./crossvalidate.pl ./music-classification`.
