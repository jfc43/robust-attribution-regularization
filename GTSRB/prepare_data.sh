wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
mkdir data
mv traffic-signs-data.zip data/
cd data
unzip traffic-signs-data.zip
cd ..
python3 generate_data.py