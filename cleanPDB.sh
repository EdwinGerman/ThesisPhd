CUR_PATH=$(pwd)
#echo "$CUR_PATH"
for file in $(ls *.pdb)
do
    #grep -v 'OT2\|OT1' $file > $file.New.pdb
    grep -v 'OT2\|OT1' $file > $CUR_PATH/pasta1/$file
done
