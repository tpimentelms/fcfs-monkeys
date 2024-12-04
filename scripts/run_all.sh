for seed in 00 01 02 03 04 05 06 07 08 09
do
    for lang in 'fi' pt id en tr he
    do
        make LANGUAGE=${lang} SEED=${seed}
    done
done
