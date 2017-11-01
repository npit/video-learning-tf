#!/usr/bin/env bash
targetdir="$1"
[  $targetdir == ""  ] && echo "Give base dir" && exit
# move to basedir
for i in $(ls  $targetdir ); do echo $i; mv $targetdir/$i/* $targetdir/; rmdir $targetdir/$i; done

# strip .avis
for i in $(ls $targetdir ); do echo $i; mv $targetdir/$i $(echo $targetdir/$i | rev | cut -c5- | rev) ; done 


