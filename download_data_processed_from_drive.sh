echo "Install gdown...."
pip install gdown
echo "Download data...."
gdown https://drive.google.com/uc?id=17e1tNVhVjMXTzM8TM5lzlM3p-MB8NzHX
echo "Unzip data...."
unzip -x data_processed.zip
echo "Success!"