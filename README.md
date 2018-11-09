# NeuBI
**Neu**ral **B**acteriocin **I**dentifier (NeuBI) is a recurrent neural network based model to predict bacteriocins from 
protein sequences. This is the software from the paper

> add citation

#### Installation

Installation of NeuBI is pretty much straight forward. You only need python 3.6. The software has only been tested for python
3.6 on a linux machine.

First, download this repository. Then run the following commands

```
$ cd NeuBI
```

We will use virtualenv to make sure the software is installed in its own sandbox, and the installations of the softwares 
needed to run NeuBI do not affect the computer's default computing environment. Run

```
$ virtualenv bacteriocin_software
```

If virtualenv is not installed, do a pip installation.

```
$ pip install virtualenv
```

Activate the newly created virtual environment.

```
$ source bacteriocin_software/bin/activate
```

Now, install the necessary softwares to run NeuBI with

```
$ pip install -r requirements.txt
```

Now, you can get NeuBI to run against any fasta file with the following commmand

```
$ python neubi.py my_fav_fasta.fa
```

The program will output a new file that has the same name + 'results'. This file will contain all the sequences from your
original fasta file that have lengths of <=302, and will assign a probability that the sequence is a bacteriocin. For example, 
a result file will have something like this

```
>EFGCFBOH_00014 Accessory gene regulator A|249|0.010913903
MNIFVLEDDFLHQTRIEKIIYKILTDNKLEVNHLEVYGKPNQLLEDISERGRHQLFFLDIDIKGEDKKGMEIAVEIRNRDPHAVIVFVTTHSEFMPVSFQYQVSALDFIDKELPEELF
SHRIEKAITYVQDNQGKTLAEDSFVFINVKSQIQVPFSDLLYIETSLIPHKLILYSTKQRVEFYGQLSEIVEQDDRLFQCHRSFVVNPYNISSIDRSERLVYLKGGLSCIVSRLKIRSLIKVV
EELHTKEK
>EFGCFBOH_00016 hypothetical protein|41|0.09664696
MNNKKTKNNFSTLSESELLKVIGGDIFKLVIDHISMKARKK
>EFGCFBOH_00019 hypothetical protein|74|0.9983411
MNTTKKQFEVIDDIKLSLMGGGSKISVGEVGQALAVCTLAGATIGSVFPIAGTLGGAVLGAQYCTGAWAIIRAH
```

Here, the number at the end in the description line is the assigned probability of a sequence being a bacteriocin. We suggest
using a threshold of >=0.9 to decide if it is a bacteriocin or not. But depending on your task, you may want the threshold 
to be flexible.

If you see an error from Theano, you might need to change the backend of Keras to Tensorflow. Theano is no longer supported 
as a deep learning library, so we are using Tensorflow. You can change this by 

```
$ cd ~/.keras
$ vim keras.json
```

You will see this inside the file

```
{                                                                                                                                                                         
"image_dim_ordering": "th",                                                                                                                                           
"epsilon": 1e-07,                                                                                                                                                     
"floatx": "float32",                                                                                                                                                  
"backend": "theano"                                                                                                                                               
}    
```

Change the backend to `tensorflow`.
