# LOVE: Learning Ontologies Via Embeddings

Food ontologies require significant effort to create and maintain, as it involves manual and time-consuming tasks. In this project, we propose a semi-supervised framework for automated ontology learning from an existing ontology scaffold by using word embeddings.

![Figure 1](/../images/Figure%201.jpg?raw=true)

## 1. Directories

Following is a short description of each directory under the root folder.

* <code>[config](./config)</code>: Contains all configurations files.
* <code>[data](./data)</code>: Contains all data files.
* <code>[hpc_scripts](./hpc_scripts)</code>: Scripts for running the code on HPC.
* <code>[managers](./managers)</code>: Contains all python modules.
* <code>[output](./output)</code>: All output files go here.
* <code>[utils](./utils)</code>: Other utility files used in the project go here.

## 2. Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 2a. Prerequisites

In addition to Python 3.6+, you can run the following command to install the required Python libraries.

```
pip install -r requirements.txt
```

Python package ```pattern``` depends on ```libmysqlclient-dev```. For Debian / Ubuntu, install like following.

```
sudo apt-get install libmysqlclient-dev
```

### 2b. Downloading Data
You need to download the GloVe 6B pre-trained weights. Following command downloads the word embeddings in GloVe format and converts them to Word2Vec compatible format.
```
cd root/data/pretrain
./download_convert_glove.sh
```

### 2c. Running

Configuration files use a general path `/path/to/project/root/directory` for compatibility. Please update these general paths to match your local computer. You can run the following script to do so.

```
# Update to local path.
./update_paths.sh

# You can optionally revert to the original path by running the following command.
./update_paths.sh revert
```

You can run all the code by running the following script. Please refer to the in-line comments of the script for details.

```
cd managers
python parse_foodon.py
cd ..
./run.sh
```

## 3. Authors

* **Jason Youn** @ [https://github.com/jasonyoun](https://github.com/jasonyoun)
* **Tarini Naravane** @ [https://github.com/nytarini](https://github.com/nytarini)

## 4. Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

## 5.Citation

Paper is under review. This section will be updated once paper is published.

## 6. License

This project is licensed under the GNU GPLv3 License. Please see the <code>[LICENSE](./LICENSE)</code> file for details.

## 7. Acknowledgments

* We would like to thank the members of the Tagkopoulos lab for their suggestions.
