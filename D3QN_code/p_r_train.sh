for gr_num in $(seq 1 1 3)
do
	for po_num in $(seq 1 1 5)
	do
		python3 train.py ${gr_num} ${po_num} &
	done
	wait
done

