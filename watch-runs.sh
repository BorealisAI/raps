
while true; do
    clear
    seq $1 $2 | xargs -L1 ts -o | xargs tail -n1 \
        | awk 'NR % 2 == 0'
    sleep 5
done
