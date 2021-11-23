srcExt=mp3
destExt=wav

srcDir=$1
destDir=$2

opts='-acodec pcm_s16le -ac 1 -ar 16000'

for filename in "$srcDir"/*.$srcExt; do

        basePath=${filename%.*}
        baseName=${basePath##*/}

        ffmpeg -i "$filename" $opts "$destDir"/"$baseName"."$destExt"

done

echo "Conversion from ${srcExt} to ${destExt} complete!"