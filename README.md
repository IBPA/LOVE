# LOVE: Learning Ontologies Via Embeddings

Food ontologies require significant effort to create and maintain, as it involves manual and time-consuming tasks. In this project, we propose a semi-supervised framework for automated ontology learning from an existing ontology scaffold by using word embeddings.

![Image of Yaktocat](https://ucdavis.box.com/s/mojo6nrokh6ad7l6ilmehibt3ipe7mmh)

## Directories

Following is a short description of each directory under the root folder.

* <code>[config](./config)</code>: Contains all configurations files.
* <code>[data](./data)</code>: Contains all data files.
* <code>[managers](./managers)</code>: Contains all python modules.
* <code>[output](./output)</code>: All output files go here.
* <code>[utils](./utils)</code>: Other utility files used in the project go here.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In addition to Python 3.6+, you can run the following command to install the required Python libraries.

```
pip install -r requirements.txt
```

Python package ```pattern``` depends on ```libmysqlclient-dev```. For Debian / Ubuntu, install like following.

```
sudo apt-get install libmysqlclient-dev
```

### Downloading Data
You also need to download some data files and place them in appropriate places.

1. GloVe 6B pre-trained weights. This downloads the word embeddings in GloVe format and converts them to Word2Vec compatible format.
```
cd root/data/pretrain
./download_convert_glove.sh
```

### Running

Configuration files use a general path `/path/to/project/root/directory` for compatibility. Please update these general paths to match your local computer. You can run the following script to do so.

```
# Update to local path.
./update_paths.sh

# You can optionally revert to the original path by running the following command.
./update_paths.sh revert
```

Following line looks up all the words found in the FDC data on WikiPedia. Only the summary section of WikiPedia is downloaded. The WikiPedia data is also preprocessed.

```
python3 parse_wikipedia.py
```

Following line trains embeddings of the words from WikiPedia.

```
python3 word2vec.py --config_file ./config/word2vec_wiki.ini
```

## Authors

* **Jason Youn** @ [https://github.com/jasonyoun](https://github.com/jasonyoun)
* **Tarini Naravane** @ [https://github.com/nytarini](https://github.com/nytarini)

## Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

## Citation

Paper is under review. This section will be updated once paper is published.

## License

This project is licensed under the GNU GPLv3 License. Please see the <code>[LICENSE.md](./LICENSE.md)</code> file for details.

## Acknowledgments

We would like to thank the members of the Tagkopoulos lab for their suggestions. TN is supported by USDA grant 58-8040-8-015, and JY is supported by the Innovation Institute for Food and Health (IIFH) pilot grant, both to IT.
