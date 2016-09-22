
README.txt

============================================================
** Contact Info 
============================================================
Sangeun Kum <keums@kaist.ac.kr>
Changheun Oh <thecow@kaist.ac.kr>
Juhan Nam <juhannam@kaist.ac.kr>

Korea Advanced Institute of Science and Technology 

============================================================
** Description 
============================================================
This is our submission to the 2016 MIREX melody extraction task.
The algorithm is a classification based approach using deep neural networks.
The file 'main.py' is the main function for calling the algorithm. 
It takes as parameter, input the full path string for the input file and output file.
If you want to know about this algorithms, 
please check https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/07/119_Paper.pdf

============================================================
** Platform and Requirements
============================================================
1. OS : LINUX 

2. Programming language : Python 2.7

3. Python Library : 
  1) Keras (Deep Learning library for Theano)
    >> http://keras.io/
  
  2) Theano (Backend of Keras)
    >> http://deeplearning.net/software/theano/install.html#install
    
  3) Librosa (for audio analysis such as laod,STFT,resampling)  
    >> http://librosa.github.io/librosa/

  4) ffmpeg 
    >> https://www.ffmpeg.org/
    >> for install : brew install ffmpeg 

  5) Numpy, SciPy

4. Hardware
  1) GPU : GeForce GTX 980 
    >> https://developer.nvidia.com/cuda-toolkit

5. Expected runtime : 2~3 seconds/song 
     
============================================================
** Use 
============================================================
The algorithm is called as follows: 

(to call from the command line)
>>python main.py <parameter> <input path> <ouput path>
ex) >>python main.py 0.2 '/home/keums/Melody/dataset/adc2004_full_set/file/pop3.wav' './SAVE_RESULTS/pop3.txt'

or

(to call from the shell)
>>main(param = 0.2, PATH_LOAD_FILE='/home/keums/Melody/dataset/adc2004_full_set/file/pop4.wav', PATH_SAVE_FILE='./SAVE_RESULTS/pop4.txt')

** default param = 0.2, 
if the voice recall rate is low, increaing the param would be effective (0 <= param <= 1 ) 


