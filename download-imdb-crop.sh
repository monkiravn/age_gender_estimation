if [[ ! -d "data" ]]
then
    mkdir "data"
fi

curl https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar -O
tar -xvf imdb_crop.tar -C data