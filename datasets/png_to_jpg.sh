cd $1
for file in *.png; do 
    mv "$file" "${file%.png}.jpg"
done