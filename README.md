# NeuBI
**Neu**ral **B**acteriocin **I**dentifier (NeuBI) is a recurrent neural network based software to predict bacteriocins from 
protein sequences. Unlike traditional alignment based approaches such as BLAST or HMMER used by BAGEL or BACTIBASE, this is an alignment free approach towards finding novel bacteriocins. We show that performance of NeuBI is better than the alignment based approaches. You can learn more from our paper:

> Md-Nafiz Hamid, Iddo Friedberg; Identifying Antimicrobial Peptides using Word Embedding with Deep Recurrent Neural Networks, Bioinformatics, bty937, https://doi.org/10.1093/bioinformatics/bty937

#### Installation

Installation of NeuBI is pretty straight forward. You only need python 3.6. The software has only been tested for python
3.6 on a linux machine.

First, download this repository. Then run the following commands

```
$ cd NeuBI
```

We will use virtualenv to make sure the software is installed in its own sandbox, and the installation of the softwares 
needed to run NeuBI do not affect the computer's default computing environment. Run

```
$ virtualenv bacteriocin_software
```

If you get an error that means virtualenv is not installed. So, do a pip installation of virtualenv.

```
$ pip install virtualenv
```

Activate the newly created virtual environment.

```
$ source bacteriocin_software/bin/activate
```

Now, based on your OS, install the necessary softwares to run NeuBI with

```
$ pip install -r requirements_linux.txt
```

For MacOS, you will need to run the previous command with `requirements_mac.txt`.

Now, you can get NeuBI to run against any fasta file with the following commmand

```
$ python neubi.py my_fav_fasta.fa
```

The program will output a new file that has the same name + 'results'. This file will contain all the sequences from your
original fasta file that have lengths of <=302, and will assign a probability that the sequence is a bacteriocin. For example, a result file will have something like this

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

Here, the number at the end of the description line is the assigned probability of a sequence being a bacteriocin. We suggest
using a threshold of >=0.95 to decide if it is a bacteriocin or not. But depending on your task, you may want the threshold 
to be flexible.

When your work is done, you can deactivate the virtual environment with the following command

```
deactivate
```

If you see an error from Theano when running NeuBI, you might need to change the backend of Keras to Tensorflow. Theano is no longer supported as a deep learning library, so we are using Tensorflow. You can change this by 

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

If you face any problem installing the software, you can contact me through email: `nafiz dot hamid dot iut at gmail dot com`
