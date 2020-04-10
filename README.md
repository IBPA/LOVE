# Project Title

Creating food ontology using unsupervised word embeddings.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Directories

Following is a short description of each directory under the root folder.
<!-- 
* <code>config/</code>: .ini configuration files go here.
* <code>data/</code>: Data to be used as an input go here.
* <code>output/</code>: All files generated from running the code will go here.
* <code>preprocess/</code>: Contains all the modules for preproessing the data.
* <code>utils/</code>: Other utility files used in the project.
 -->
### Prerequisites

In addition to Python 3.6+, following Python libraries are required.

```
numpy==1.16.3
pandas==0.24.2
matplotlib==3.1.0
scikit-learn==0.20.0
scipy==1.3.1
argparse==1.1
configparser==3.5.0
gensim==3.8.1
cython==0.29.14
pattern==3.6
wikipedia==1.4.0
```

You can optionally use pip3 to install the required Python libraries.

```
pip3 install -r requirements.txt
```

Python package ```pattern``` depends on ```libmysqlclient-dev```. For Debian / Ubuntu, install like following.

```
sudo apt-get install libmysqlclient-dev
```

### Downloading Data
You also need to download some data files and place them in appropriate places.

1. FDC data.
```
cd ./data/FDC
./download_fdc.sh
```

2. GloVe 6B pre-trained weights. This downloads the word embeddings in GloVe format and converts them to Word2Vec compatible format.
```
cd ./data/pretrain
./download_convert_glove.sh
```

3. (Optional) Word2Vec models and the actual word embeddings. Use the following password when prompted: sc!@#$
```
cd ./data/model
scp jyoun@tagkopouloslab.ucdavis.edu:/home/jyoun/FoodData/model.zip .
unzip model.zip
```

4. (Optional) Output files. se the following password when prompted: sc!@#$
```
cd ./output
scp jyoun@tagkopouloslab.ucdavis.edu:/home/jyoun/FoodData/output_files.zip .
unzip output_files.zip
```

### Running

Configuration files use a general path ```/path/to/project/root/directory``` for compatibility. Please update these general paths to match your local computer. You can run the following script to do so.

```
./update_paths.sh
```

Following line loads the FDC data and preprocesses it.

```
python3 prepare_data.py
```

Following line looks up all the words found in the FDC data on WikiPedia. Only the summary section of WikiPedia is downloaded. The WikiPedia data is also preprocessed.

```
python3 parse_wikipedia.py
```

Following line trains embeddings of the words from WikiPedia.

```
python3 word2vec.py --config_file ./config/word2vec_wiki.ini
```

Following line trains embeddings of the words from FDC.

```
python3 word2vec.py --config_file ./config/word2vec_fdc.ini
```

### Results

What are the output files?

## Authors

* **Jason Youn** - *Initial work* - [https://github.com/jasonyoun](https://github.com/jasonyoun)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc